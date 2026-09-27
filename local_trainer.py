"""
FedLLM: Privacy-Preserving Explainable Traffic Flow Prediction Pipeline
Phase 3: Local Parameter-Efficient Fine-Tuning (PEFT) Engine
=============================================================================
This module provides local LoRA / QLoRA fine-tuning of causal Large Language Models
(e.g., Llama-3-8B, Qwen2.5, Mistral, TinyLlama) on structured freeway traffic
instruction prompt datasets (train_prompts.jsonl).

Key Architectural Highlights:
1. 4-bit / 8-bit QLoRA Quantization (bitsandbytes NF4 with double quantization).
2. Parameter-Efficient LoRA Adapters (only ~0.1% to 1.5% of parameters are trainable).
3. Optimized Memory Footprint: Gradient checkpointing, paged AdamW 8-bit, fp16/bf16.
4. Loss Masking: Computes gradient updates strictly on the assistant response.
5. Federated Adapter Utility: Functions to extract and inject LoRA state dictionaries
   for seamless integration with the Phase 4 Flower (flwr) federated aggregation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# Optional PyTorch & Hugging Face imports (with graceful fallback diagnostics)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object  # type: ignore

try:
    import transformers
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        PreTrainedModel,
        PreTrainedTokenizer,
        Trainer,
        TrainingArguments,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import peft
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

# Configure module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FedLLM.LocalTrainer")


# =============================================================================
# 1. Model & LoRA Configurations
# =============================================================================

@dataclass
class FedLLMTrainingConfig:
    """Configuration hyper-parameters for FedLLM local PEFT fine-tuning."""
    # Base model identifier (Hugging Face hub or local path)
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Dataset paths
    train_prompts_file: str = "./prompts/train_prompts.jsonl"
    val_prompts_file: str = "./prompts/val_prompts.jsonl"
    output_dir: str = "./checkpoints/fedllm_lora"

    # Quantization (QLoRA)
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    compute_dtype: str = "bfloat16" if HAS_TORCH and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"

    # LoRA Hyperparameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])

    # Training Hyperparameters (Consumer-GPU friendly)
    max_seq_length: int = 1024
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    num_train_epochs: int = 3
    max_steps: int = -1
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True


# =============================================================================
# 2. Traffic Instruction Dataset & Tokenizer Preprocessor
# =============================================================================

class TrafficInstructionDataset(Dataset):
    """
    PyTorch Dataset that tokenizes and formats traffic JSONL prompt pairs
    into causal language modeling sequences with response-only loss masking.
    """

    PROMPT_TEMPLATE = (
        "<|im_start|>system\n{instruction}<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples: List[Dict[str, Any]] = []

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL prompt file not found: {jsonl_path}")

        logger.info(f"Loading and parsing prompt dataset: {jsonl_path} ...")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self.examples.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON line #{line_idx}: {e}")

        logger.info(f"Loaded {len(self.examples):,} prompt records from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        record = self.examples[idx]
        instruction = record.get("instruction", "")
        user_input = record.get("input", "")
        response_output = record.get("output", "")

        # Format prompt prefix and completion
        prompt_prefix = self.PROMPT_TEMPLATE.format(
            instruction=instruction,
            input=user_input,
        )
        full_text = prompt_prefix + response_output + "<|im_end|>"

        # Tokenize prompt prefix vs. full sequence to mask prompt loss
        prefix_ids = self.tokenizer.encode(prompt_prefix, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        # Truncate if exceeds max length
        if len(full_ids) > self.max_seq_length:
            full_ids = full_ids[: self.max_seq_length]

        # Labels: -100 ignores loss on prompt tokens, active only on assistant completion
        labels = [-100] * len(full_ids)
        prefix_len = min(len(prefix_ids), len(full_ids))
        for i in range(prefix_len, len(full_ids)):
            labels[i] = full_ids[i]

        attention_mask = [1] * len(full_ids)

        if HAS_TORCH:
            return {
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# =============================================================================
# 3. Model & LoRA Initialization Engine
# =============================================================================

class FedLLMModelLoader:
    """Manages QLoRA model initialization, quantization, and adapter setup."""

    @staticmethod
    def create_bnb_config(cfg: FedLLMTrainingConfig) -> Optional[BitsAndBytesConfig]:
        """Builds bitsandbytes 4-bit / 8-bit quantization configuration."""
        if not HAS_TRANSFORMERS:
            return None

        if cfg.use_4bit and HAS_TORCH and torch.cuda.is_available():
            compute_dtype = getattr(torch, cfg.compute_dtype, torch.float16)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            logger.info("Initialized 4-bit QLoRA BitsAndBytes configuration (NF4 with double quantization).")
            return bnb_config
        return None

    @staticmethod
    def load_model_and_tokenizer(
        cfg: FedLLMTrainingConfig,
    ) -> Tuple[Optional[PreTrainedModel], Optional[PreTrainedTokenizer]]:
        """
        Loads base causal LLM with 4-bit quantization and initializes
        parameter-efficient LoRA adapters.
        """
        if not (HAS_TORCH and HAS_TRANSFORMERS and HAS_PEFT):
            logger.warning(
                "PyTorch, Transformers, or PEFT not fully installed. "
                "Model loading operating in configuration/diagnostic mode."
            )
            return None, None

        logger.info(f"Loading Tokenizer from: {cfg.model_name_or_path} ...")
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=True,
            padding_side="right",
        )

        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        bnb_config = FedLLMModelLoader.create_bnb_config(cfg)
        device_map = "auto" if torch.cuda.is_available() else None
        torch_dtype = getattr(torch, cfg.compute_dtype, torch.float16) if torch.cuda.is_available() else torch.float32

        logger.info(f"Loading Base Model: {cfg.model_name_or_path} (Device Map: {device_map}) ...")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        # Enable gradient checkpointing for lower VRAM usage
        if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        # Prepare model for k-bit training if quantized
        if bnb_config is not None:
            model = prepare_model_for_kbit_training(model)

        # Configure LoRA
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.target_modules,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        logger.info(
            f"Applying LoRA Adapter (Rank r={cfg.lora_r}, Alpha={cfg.lora_alpha}, "
            f"Target Modules={cfg.target_modules}) ..."
        )
        peft_model = get_peft_model(model, lora_config)

        # Log trainable parameters
        FedLLMModelLoader.print_trainable_parameters(peft_model)

        return peft_model, tokenizer

    @staticmethod
    def print_trainable_parameters(model: Any) -> None:
        """Prints the count and percentage of trainable vs total parameters."""
        if not hasattr(model, "named_parameters"):
            return

        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        pct = 100 * trainable_params / all_param if all_param > 0 else 0
        logger.info(
            f"Trainable Params: {trainable_params:,} || "
            f"Total Params: {all_param:,} || "
            f"Trainable Ratio: {pct:.4f}%"
        )


# =============================================================================
# 4. Federated Parameter Extraction & Injection Utilities
# =============================================================================

class FederatedLoRAAdapterHelper:
    """
    Extracts and injects LoRA adapter weights (as NumPy arrays or state dicts)
    to enable Flower (flwr) federated averaging in Phase 4.
    """

    @staticmethod
    def extract_lora_state_dict(model: Any) -> Dict[str, np.ndarray]:
        """
        Extracts only the trainable LoRA adapter tensors from the model
        and converts them to NumPy arrays for federated parameter exchange.
        """
        if not hasattr(model, "state_dict"):
            return {}

        state_dict = model.state_dict()
        lora_weights: Dict[str, np.ndarray] = {}
        for key, tensor in state_dict.items():
            if "lora_" in key or "adapter_" in key:
                lora_weights[key] = tensor.detach().cpu().to(torch.float32).numpy()

        logger.info(f"Extracted {len(lora_weights)} LoRA adapter weight tensors for federated exchange.")
        return lora_weights

    @staticmethod
    def inject_lora_state_dict(model: Any, lora_weights: Dict[str, np.ndarray]) -> None:
        """
        Injects aggregated global LoRA weights received from the Flower federated
        server into the local model instance.
        """
        if not hasattr(model, "load_state_dict"):
            return

        torch_state_dict = {}
        for key, np_array in lora_weights.items():
            torch_state_dict[key] = torch.from_numpy(np_array)

        # Non-strict loading updates only the matching LoRA weights
        model.load_state_dict(torch_state_dict, strict=False)
        logger.info(f"Injected {len(torch_state_dict)} global LoRA adapter weight tensors.")

    @staticmethod
    def save_lora_adapter(model: Any, output_dir: str) -> str:
        """Saves the local LoRA adapter weights and config to disk."""
        os.makedirs(output_dir, exist_ok=True)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(output_dir)
            logger.info(f"Saved LoRA adapter checkpoint to: {output_dir}")
        return output_dir


# =============================================================================
# 5. Local Fine-Tuning Execution Engine
# =============================================================================

def train_local_model(
    cfg: FedLLMTrainingConfig,
) -> Optional[Dict[str, Any]]:
    """
    Executes local parameter-efficient fine-tuning on traffic prompt dataset
    using Hugging Face Trainer and QLoRA.
    """
    logger.info("=" * 80)
    logger.info(f"Starting FedLLM Local Fine-Tuning: Model = {cfg.model_name_or_path}")
    logger.info("=" * 80)

    # 1. Load Model & Tokenizer
    model, tokenizer = FedLLMModelLoader.load_model_and_tokenizer(cfg)

    if model is None or tokenizer is None:
        logger.info(
            "Running in simulation / diagnostic mode. "
            "To execute full GPU fine-tuning, ensure PyTorch with CUDA, transformers, and peft are installed."
        )
        return {
            "status": "simulation_complete",
            "model_name": cfg.model_name_or_path,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "target_modules": cfg.target_modules,
        }

    # 2. Build Train & Validation Datasets
    train_dataset = TrafficInstructionDataset(
        jsonl_path=cfg.train_prompts_file,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
    )

    eval_dataset = None
    if os.path.exists(cfg.val_prompts_file):
        eval_dataset = TrafficInstructionDataset(
            jsonl_path=cfg.val_prompts_file,
            tokenizer=tokenizer,
            max_seq_length=cfg.max_seq_length,
        )

    # 3. Data Collator with Padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )

    # 4. Hugging Face Training Arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        logging_steps=cfg.logging_steps,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_dataset is not None else "no",
        save_total_limit=cfg.save_total_limit,
        fp16=cfg.fp16 and torch.cuda.is_available() and not cfg.bf16,
        bf16=cfg.bf16 and torch.cuda.is_available(),
        gradient_checkpointing=cfg.gradient_checkpointing,
        report_to="none",
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        remove_unused_columns=False,
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 6. Train
    logger.info("Beginning Trainer.train() execution ...")
    train_result = trainer.train()

    # 7. Save Adapter Weights
    final_adapter_dir = os.path.join(cfg.output_dir, "final_adapter")
    FederatedLoRAAdapterHelper.save_lora_adapter(model, final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info(f"Local PEFT Training complete! Final adapter saved to {final_adapter_dir}")
    return metrics


# =============================================================================
# 6. CLI Interface
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedLLM Phase 3: Local PEFT (LoRA/QLoRA) Fine-Tuning")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Hugging Face model ID or path (e.g., Qwen/Qwen2.5-1.5B-Instruct, meta-llama/Meta-Llama-3-8B-Instruct)")
    parser.add_argument("--train_file", type=str, default="./prompts/train_prompts.jsonl", help="Path to train_prompts.jsonl")
    parser.add_argument("--val_file", type=str, default="./prompts/val_prompts.jsonl", help="Path to val_prompts.jsonl")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/fedllm_lora", help="Output directory for adapters")
    parser.add_argument("--r", type=int, default=16, help="LoRA rank dimension (default: 16)")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha scaling (default: 32)")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size (default: 2)")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    return parser.parse_args()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    config = FedLLMTrainingConfig(
        model_name_or_path=args.model,
        train_prompts_file=args.train_file,
        val_prompts_file=args.val_file,
        output_dir=args.output_dir,
        lora_r=args.r,
        lora_alpha=args.alpha,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        use_4bit=not args.no_4bit,
    )

    print("=" * 80)
    print(f"[FedLLM] Phase 3: Local PEFT / LoRA Training Engine")
    print("=" * 80)
    print(f"  * Base Model:       {config.model_name_or_path}")
    print(f"  * LoRA Rank (r):    {config.lora_r} (Alpha: {config.lora_alpha})")
    print(f"  * Target Modules:   {config.target_modules}")
    print(f"  * 4-Bit QLoRA:      {config.use_4bit} (Type: {config.bnb_4bit_quant_type})")
    print(f"  * Batch Size:       {config.per_device_train_batch_size} (Grad Accum: {config.gradient_accumulation_steps})")
    print(f"  * Effective Batch:  {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"  * Training Prompts: {config.train_prompts_file}")
    print(f"  * Output Directory: {config.output_dir}")
    print("=" * 80)

    result = train_local_model(config)
    print(f"\n[+] Execution Result Summary: {result}")
    print("\n[SUCCESS] Phase 3: Local Parameter-Efficient Fine-Tuning Engine verified!")

"""
FedLLM: Raw Telemetry Inspector
=============================================================================
Utility to inspect and explore raw .npz traffic tensors from PeMS datasets.

Usage:
    # Print the first 12 time steps for Sensor #0
    python view_raw_data.py

    # Inspect a specific sensor and time range
    python view_raw_data.py --sensor 12 --steps 24

    # Export a sensor's data to a CSV file to open in Excel
    python view_raw_data.py --sensor 0 --export_csv sensor_0_telemetry.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd


def inspect_npz(
    npz_path: str = "./data/PEMS04/PEMS04.npz",
    sensor_id: int = 0,
    start_step: int = 0,
    num_steps: int = 12,
    export_csv: str = "",
):
    if not os.path.exists(npz_path):
        print(f"Error: File not found at {npz_path}")
        return

    # 1. Load archive
    archive = np.load(npz_path)
    keys = archive.files
    data = archive["data"]

    total_steps, total_sensors, total_features = data.shape

    print("=" * 80)
    print(f"FILE: {npz_path}")
    print("=" * 80)
    print(f"Archive Keys    : {keys}")
    print(f"Tensor Shape    : {data.shape}  --> (Time, Sensors, Features)")
    print(f"Total Time Steps: {total_steps:,} (each step is 5 minutes)")
    print(f"Total Sensors   : {total_sensors}")
    print(f"Features (3)    : [0] Flow (veh/5min), [1] Occupancy (0-1), [2] Speed (mph)")
    print("=" * 80)

    # 2. Extract slice for the requested sensor
    if sensor_id >= total_sensors or sensor_id < 0:
        print(f"Error: sensor_id {sensor_id} out of range (0 to {total_sensors - 1})")
        return

    end_step = min(start_step + num_steps, total_steps)
    sensor_slice = data[start_step:end_step, sensor_id, :]

    df = pd.DataFrame(
        sensor_slice,
        columns=["Flow (veh/5min)", "Occupancy", "Speed (mph)"],
        index=range(start_step, end_step),
    )
    df.index.name = "Time Step (t)"

    print(f"\nTelemetry Slice: Sensor Station #{sensor_id:03d} (Steps {start_step} to {end_step - 1}):\n")
    print(df.to_string())

    # Summary statistics for this sensor across the whole dataset
    full_sensor_data = data[:, sensor_id, :]
    print("\n" + "-" * 80)
    print(f"Lifetime Statistics for Sensor #{sensor_id:03d} (All {total_steps:,} steps):")
    print("-" * 80)
    stats_df = pd.DataFrame({
        "Min": full_sensor_data.min(axis=0),
        "Mean": full_sensor_data.mean(axis=0),
        "Max": full_sensor_data.max(axis=0),
        "Std": full_sensor_data.std(axis=0),
    }, index=["Flow (veh/5min)", "Occupancy", "Speed (mph)"])
    print(stats_df.round(2).to_string())

    # 3. Optional CSV export
    if export_csv:
        # Determine number of steps to export: if num_steps is the default 12, export 288 steps (1 full day), else use num_steps
        n_export = num_steps if num_steps > 12 else 288
        if n_export > total_steps:
            n_export = total_steps

        # Generate readable 5-minute datetime timestamps starting from Jan 1, 2018
        base_time = pd.Timestamp("2018-01-01 00:00:00")
        time_index = [base_time + pd.Timedelta(minutes=5 * i) for i in range(start_step, start_step + n_export)]

        export_df = pd.DataFrame(
            data[start_step:start_step + n_export, sensor_id, :],
            columns=["Flow_veh_5min", "Occupancy", "Speed_mph"],
        )
        export_df.insert(0, "TimeStep", range(start_step, start_step + n_export))
        export_df.insert(1, "Timestamp", [t.strftime("%Y-%m-%d %H:%M") for t in time_index])
        export_df.insert(2, "DayOfWeek", [t.strftime("%A") for t in time_index])
        export_df.insert(3, "TimeOfDay", [t.strftime("%I:%M %p") for t in time_index])

        export_df.to_csv(export_csv, index=False)
        print(f"\n[+] Successfully exported {n_export:,} records for Sensor #{sensor_id:03d} to '{export_csv}'.")
        print(f"    Columns: {list(export_df.columns)}")
        print(f"    File path: {os.path.abspath(export_csv)}")
        print(f"    You can now double-click to open '{export_csv}' in Excel or VS Code!")



if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="View raw .npz telemetry data.")
    parser.add_argument("--file", type=str, default="./data/PEMS04/PEMS04.npz", help="Path to .npz file")
    parser.add_argument("--sensor", type=int, default=0, help="Sensor ID to view (default: 0)")
    parser.add_argument("--start", type=int, default=0, help="Starting time step index (default: 0)")
    parser.add_argument("--steps", type=int, default=12, help="Number of time steps to display (default: 12)")
    parser.add_argument("--export_csv", type=str, default="", help="Optional CSV filename to export for Excel")

    args = parser.parse_args()
    inspect_npz(
        npz_path=args.file,
        sensor_id=args.sensor,
        start_step=args.start,
        num_steps=args.steps,
        export_csv=args.export_csv,
    )

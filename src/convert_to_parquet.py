"""
Convert UNSW-NB15 CSV datasets to Parquet format for faster I/O.
Run this script first before training or launching the Streamlit app.

Usage:
    python src/convert_to_parquet.py
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

FILES = {
    "UNSW_NB15_training-set.csv": "UNSW_NB15_training-set.parquet",
    "UNSW_NB15_testing-set.csv": "UNSW_NB15_testing-set.parquet",
}


def convert():
    for csv_name, parquet_name in FILES.items():
        csv_path = os.path.join(DATA_DIR, csv_name)
        parquet_path = os.path.join(DATA_DIR, parquet_name)

        if not os.path.exists(csv_path):
            print(f"[SKIP] {csv_name} not found in {DATA_DIR}")
            continue

        print(f"[READ] {csv_name} ...", end=" ")
        df = pd.read_csv(csv_path)

        # Clean leading/trailing whitespace in attack_cat
        if "attack_cat" in df.columns:
            df["attack_cat"] = df["attack_cat"].astype(str).str.strip()

        df.to_parquet(parquet_path, engine="pyarrow", index=False)

        csv_size = os.path.getsize(csv_path) / (1024 * 1024)
        parquet_size = os.path.getsize(parquet_path) / (1024 * 1024)
        ratio = csv_size / parquet_size

        print(
            f"CSV: {csv_size:.1f} MB -> Parquet: {parquet_size:.1f} MB "
            f"({ratio:.1f}x smaller)"
        )

    print("\n[DONE] Parquet conversion complete.")


if __name__ == "__main__":
    convert()

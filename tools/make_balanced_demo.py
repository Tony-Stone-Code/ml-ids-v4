import argparse
from pathlib import Path

import pandas as pd

DEFAULT_SIZES = [200, 500, 1000, 2000, 5000]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a balanced demo subset from the official test dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV path (e.g., data/UNSW_NB15_testing-set.csv)",
    )
    parser.add_argument(
        "--total-size",
        type=int,
        help="Total number of rows in a single balanced subset",
    )
    parser.add_argument(
        "--total-sizes",
        help="Comma-separated list of total sizes (e.g., 200,1000,5000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--class-col",
        default="attack_cat",
        help="Column name to balance on (default: attack_cat)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/demo_samples",
        help="Output directory for generated subsets",
    )
    parser.add_argument(
        "--allow-upsample",
        action="store_true",
        help="Allow upsampling with replacement when a class is small",
    )
    return parser.parse_args()


def compute_targets(classes, total_size):
    if total_size < len(classes):
        raise ValueError(
            "total_size must be at least the number of classes for balancing"
        )
    base = total_size // len(classes)
    remainder = total_size % len(classes)
    targets = {}
    for idx, cls in enumerate(classes):
        targets[cls] = base + (1 if idx < remainder else 0)
    return targets


def sample_class(df_class, n, seed, allow_upsample):
    if n == 0:
        return df_class.head(0)
    if n <= len(df_class):
        return df_class.sample(n=n, random_state=seed, replace=False)
    if allow_upsample:
        return df_class.sample(n=n, random_state=seed, replace=True)
    return df_class.sample(n=len(df_class), random_state=seed, replace=False)


def main():
    args = parse_args()
    df = pd.read_csv(args.input, low_memory=False)

    if args.class_col not in df.columns:
        raise ValueError(f"Class column not found: {args.class_col}")

    sizes = []
    if args.total_sizes:
        for token in args.total_sizes.split(","):
            token = token.strip()
            if token:
                sizes.append(int(token))
    if args.total_size:
        sizes.append(int(args.total_size))

    if not sizes:
        sizes = DEFAULT_SIZES
        print(f"[INFO] Using default sizes: {sizes}")

    sizes = sorted(set(sizes))
    classes = sorted(df[args.class_col].astype(str).unique().tolist())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    in_stem = Path(args.input).stem

    for total_size in sizes:
        targets = compute_targets(classes, total_size)
        sampled_frames = []
        for idx, cls in enumerate(classes):
            df_cls = df[df[args.class_col].astype(str) == cls]
            n = targets[cls]
            sampled = sample_class(
                df_cls,
                n,
                seed=args.seed + idx,
                allow_upsample=args.allow_upsample,
            )
            sampled_frames.append(sampled)

        demo_df = pd.concat(sampled_frames, ignore_index=True)
        demo_df = demo_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

        out_path = out_dir / f"{in_stem}_balanced_{total_size}_seed{args.seed}.csv"
        demo_df.to_csv(out_path, index=False)

        dist = demo_df[args.class_col].value_counts().sort_index()
        print("Saved:", out_path)
        print("Class distribution:\n", dist.to_string())


if __name__ == "__main__":
    main()

import argparse
import json
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate random train/val/test splits."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        required=True,
        help="Total number of samples (0 to N-1)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="splits.json",
        help="Output JSON file name."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducibility."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    N = args.num_samples

    # All indices
    indices = np.arange(N)

    np.random.shuffle(indices)

    # 70% / 15% / 15%
    n_train = int(0.70 * N)
    n_val = int(0.15 * N)
    n_test = N - n_train - n_val 

    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:n_train + n_val].tolist()
    test_idx = indices[n_train + n_val:].tolist()

    split = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    with open(args.output, "w") as f:
        json.dump(split, f, indent=2)

    print(f"Saved split file to {args.output}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

if __name__ == "__main__":
    main()
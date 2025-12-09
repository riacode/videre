import argparse
import json
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, required=True,
                        help="Path to y_videos.npy")
    parser.add_argument("--output", type=str, default="splits.json")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main():
    args = parse_args()
    y = np.load(args.labels)   # (num_videos,)
    N = len(y)

    # First split: Train+Val vs Test (15%)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    train_val_idx, test_idx = next(sss1.split(np.zeros(N), y))

    # Second split: Train vs Val (15% of remaining = 0.1765 of total)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.1765, random_state=args.seed)
    train_idx, val_idx = next(sss2.split(np.zeros(len(train_val_idx)), y[train_val_idx]))

    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    split = {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    }

    with open(args.output, "w") as f:
        json.dump(split, f, indent=2)

    print("Train labels:", np.unique(y[train_idx], return_counts=True))
    print("Val labels:",   np.unique(y[val_idx],   return_counts=True))
    print("Test labels:",  np.unique(y[test_idx],  return_counts=True))


if __name__ == "__main__":
    main()

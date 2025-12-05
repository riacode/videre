"""
Utility script to consolidate raw train/val/test .npy files into the layout expected by scripts/train.py
"""
import json
import os
import numpy as np

def load_split(path: str):
    """Load a split .npy file and return features (without last col) and labels (last col)."""
    array = np.load(path)
    features = array[:, :-1].astype(np.float32)
    labels = array[:, -1].astype(int)
    return features, labels

def main() -> None:
    project_root = os.path.abspath(".")
    feature_dir = os.path.join(project_root, "features", "v1")
    splits_dir = os.path.join(project_root, "data", "splits")
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    # Hard-coded paths for the raw data
    train_X, train_y = load_split(os.path.join(project_root, "data", "train.npy"))
    val_X, val_y = load_split(os.path.join(project_root, "data", "val.npy"))
    test_path = os.path.join(project_root, "data", "test.npy")

    X_blocks = [train_X, val_X]
    y_blocks = [train_y, val_y]
    split_dict = {
        "train": list(range(len(train_y))),
        "val": list(range(len(train_y), len(train_y) + len(val_y))),
    }

    meta = {
        "feature_dim": int(train_X.shape[1]),
        "train_samples": int(len(train_y)),
        "val_samples": int(len(val_y)),
        "label_description": {"0": "real", "1": "fake"},
    }

    # test data
    if os.path.exists(test_path):
        test_X, test_y = load_split(test_path)
        start_idx = len(train_y) + len(val_y)
        split_dict["test"] = list(range(start_idx, start_idx + len(test_y)))
        X_blocks.append(test_X)
        y_blocks.append(test_y)
        meta["test_samples"] = int(len(test_y))

    # Save X, y
    X = np.concatenate(X_blocks, axis=0)
    y = np.concatenate(y_blocks, axis=0)
    np.save(os.path.join(feature_dir, "X.npy"), X)
    np.save(os.path.join(feature_dir, "y.npy"), y)

    # Save meta
    with open(os.path.join(feature_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save split indices
    split_path = os.path.join(splits_dir, "default.json")
    with open(split_path, "w") as f:
        json.dump(split_dict, f, indent=2)


if __name__ == "__main__":
    main()


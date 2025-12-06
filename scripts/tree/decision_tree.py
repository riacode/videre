import pandas as pd
import numpy as np
import json
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train sklearn DecisionTreeClassifier on patch embeddings")

    parser.add_argument("--feature_dir", type=str, required=True,
                        help="Directory containing X.npy, y.npy, and meta.json")
    parser.add_argument("--split_file", type=str, required=True,
                        help="Path to JSON containing train/val/test indices")
    parser.add_argument("--max_depth", type=int, default=5,
                        help="Max depth for DecisionTreeClassifier")
    parser.add_argument("--output", type=str, default="clf.pkl",
                        help="Where to save the trained model")

    return parser.parse_args()

def load_data(feature_dir):
    X = np.load(f"{feature_dir}/X.npy", mmap_mode="r")
    y = np.load(f"{feature_dir}/y.npy", mmap_mode="r")
    return X, y

def load_splits(split_file):
    with open(split_file, "r") as f:
        split = json.load(f)

    return (
        np.array(split["train"]),
        np.array(split["val"]) if "val" in split else None,
        np.array(split["test"])
    )

def extract_subset(X, y, idx):
    X_subset = X[idx]
    y_subset = y[idx]
    return X_subset, y_subset

def flatten_features(X):
    return X.reshape(len(X), -1)

def train_classifier(X_train, y_train, max_depth):
    clf = DecisionTreeClassifier(
    criterion='gini', 
    splitter='best', 
    max_depth=8, 
    min_samples_split=2, 
    min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, 
    max_features=None, 
    random_state=0, 
    max_leaf_nodes=None, 
    min_impurity_decrease=0.0, 
    class_weight=None, 
    ccp_alpha=0.0, 
    monotonic_cst=None)

    clf.fit(X_train, y_train)
    return clf

def evaluate_classifier(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))


def main():
    args = parse_args()
    X, y = load_data(args.feature_dir)
    train_idxs, val_idxs, test_idxs = load_splits(args.split_file)

    X_train, y_train = extract_subset(X, y, train_idxs)
    X_train = flatten_features(X_train)

    clf = train_classifier(X_train, y_train, 8)

   


if __name__ == "__main__":
    main()



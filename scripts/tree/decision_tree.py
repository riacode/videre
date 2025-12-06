
import argparse
import numpy as np
import json
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import joblib

# -----------------------------
# Argument parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train DecisionTree on video-level embeddings")

    parser.add_argument("--feature_dir", type=str, required=True,
                        help="Directory containing X.npy and y.npy (flattened frame-level data)")
    parser.add_argument("--real_processed", type=str, required=True,
                        help="Directory of real processed videos (.npy per video)")
    parser.add_argument("--fake_processed", type=str, required=True,
                        help="Directory of fake processed videos (.npy per video)")
    parser.add_argument("--split_file", type=str, required=True,
                        help="JSON containing train/val/test indices (video level)")
    parser.add_argument("--max_depth", type=int, default=8,
                        help="Max depth for Decision Tree")
    parser.add_argument("--output", type=str, default='/home/irisxu/videre/scripts/tree/clf_dino.pkl',
                        help="Where to save trained model (.pkl)")

    return parser.parse_args()

# -----------------------------
# Load flattened X.npy and y.npy
# -----------------------------
def load_flat_data(feature_dir):
    X = np.load(f"{feature_dir}/resnet_per_frame_X.npy", mmap_mode="r")   # (total_frames, 2048)
    y = np.load(f"{feature_dir}/resnet_per_frame_y.npy", mmap_mode="r")   # (total_frames,)
    return X, y

# -----------------------------
# Load video-level split indices
# -----------------------------
def load_splits(split_file):
    with open(split_file, "r") as f:
        split = json.load(f)

    return (
        np.array(split["train"]),
        np.array(split["val"]),
        np.array(split["test"])
    )

# -----------------------------
# Compute frames per video
# -----------------------------
def compute_frames_per_video(real_dir, fake_dir):
    frame_counts = []

    real_files = sorted([f for f in os.listdir(real_dir) if f.endswith(".npy")])
    fake_files = sorted([f for f in os.listdir(fake_dir) if f.endswith(".npy")])

    # Real
    for fname in tqdm(real_files, desc="REAL frames"):
        frames = np.load(os.path.join(real_dir, fname))
        frame_counts.append(frames.shape[0])

    # Fake
    for fname in tqdm(fake_files, desc="FAKE frames"):
        frames = np.load(os.path.join(fake_dir, fname))
        frame_counts.append(frames.shape[0])

    return frame_counts

# -----------------------------
# Reconstruct videos from flat arrays
# -----------------------------
def rebuild_videos_from_flat(X_flat, y_flat, frames_per_video):
    X_videos = []
    y_videos = []

    idx = 0
    for T in frames_per_video:
        X_videos.append(X_flat[idx:idx+T])        # (T, 2048)
        y_videos.append(int(y_flat[idx:idx+T].mean() > 0.5))
        idx += T

    return np.array(X_videos, dtype=object), np.array(y_videos)

# -----------------------------
# Train Decision Tree on frame-level samples
# -----------------------------
def train_classifier(X_train, y_train, max_depth):
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
    clf.fit(X_train, y_train)
    return clf

# -----------------------------
# Predict a single video label
# -----------------------------
def predict_video_label(clf, video_frames, threshold=0.5):
    probs = clf.predict_proba(video_frames)[:, 1]
    preds = (probs > threshold).astype(int)
    return int(preds.mean() > 0.5)

# -----------------------------
# Compute video-level accuracy
# -----------------------------
def compute_video_accuracy(clf, X_videos, y_videos, indices):
    preds = []
    labels = []
    for idx in indices:
        pred = predict_video_label(clf, X_videos[idx])
        preds.append(pred)
        labels.append(y_videos[idx])
    return accuracy_score(labels, preds)

# -----------------------------
# MAIN
# -----------------------------
def main():
    args = parse_args()

    # 1. Load flattened data
    X_flat, y_flat = load_flat_data(args.feature_dir)

    # 2. Compute how many frames each video has
    frames_per_video = compute_frames_per_video(args.real_processed, args.fake_processed)

    # 3. Reassemble video-level arrays
    X_videos, y_videos = rebuild_videos_from_flat(X_flat, y_flat, frames_per_video)

    # 4. Load split indices
    train_idxs, val_idxs, test_idxs = load_splits(args.split_file)

    # 5. Build frame-level train set
    X_train = np.concatenate([X_videos[i] for i in train_idxs], axis=0)
    y_train = np.concatenate([np.repeat(y_videos[i], len(X_videos[i])) for i in train_idxs])

    print("Training shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    # 6. Train clas
    clf = train_classifier(X_train, y_train, args.max_depth)

    # Save classifier
    joblib.dump(clf, args.output)
    print(f"Saved trained model to {args.output}")

    # -----------------------------
    # 7. Compute validation / test accuracy
    # -----------------------------
    val_acc = compute_video_accuracy(clf, X_videos, y_videos, val_idxs)
    test_acc = compute_video_accuracy(clf, X_videos, y_videos, test_idxs)

    print("\n=== RESULTS ===")
    print("Validation accuracy:", val_acc)
    print("Test accuracy:", test_acc)
if __name__ == "__main__":
    main()

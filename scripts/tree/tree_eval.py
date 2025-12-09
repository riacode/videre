import sys
import os

# Get absolute path to the videre directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

sys.path.append(BASE_DIR)

from videre.evals.metrics import compute_metrics
from sklearn.metrics import accuracy_score

import numpy as np
import json
import joblib
import os
from tqdm import tqdm


def load_splits(split_file):
    with open(split_file, "r") as f:
        split = json.load(f)
    return np.array(split["train"]), np.array(split["val"]), np.array(split["test"])


def compute_frames_per_video(real_dir, fake_dir):
    frame_counts = []
    real_files = sorted([f for f in os.listdir(real_dir) if f.endswith(".npy")])
    fake_files = sorted([f for f in os.listdir(fake_dir) if f.endswith(".npy")])
    for f in real_files:
        frame_counts.append(np.load(os.path.join(real_dir, f)).shape[0])
    for f in fake_files:
        frame_counts.append(np.load(os.path.join(fake_dir, f)).shape[0])
    return frame_counts


def rebuild_videos(X_flat, y_flat, frames_per_video):
    X_videos, y_videos = [], []
    idx = 0
    for T in frames_per_video:
        X_videos.append(X_flat[idx:idx+T])
        y_videos.append(int(y_flat[idx:idx+T].mean() > 0.5))
        idx += T
    return np.array(X_videos, dtype=object), np.array(y_videos)


def evaluate_video_with_metrics(clf, X_videos, y_videos, test_idx):
    video_preds = []
    video_true = []
    video_probas = []

    for i in test_idx:
        frames = X_videos[i]

        if len(frames) == 0:
            print(f"video {i} has 0 frames. forcing prob=0 and pred=0")
            video_true.append(y_videos[i])
            video_preds.append(0)
            video_probas.append([1.0, 0.0])
            continue
        # frame-level prob for class 1
        frame_p1 = clf.predict_proba(frames)[:, 1]

        # aggregate to video-level prob
        video_p1 = frame_p1.mean()
        video_pred = int(video_p1 > 0.5)

        video_preds.append(video_pred)
        video_true.append(y_videos[i])
        video_probas.append([1 - video_p1, video_p1])

    return (
        np.array(video_true),
        np.array(video_preds),
        np.array(video_probas)
    )


# -------------- MAIN --------------
clf = joblib.load("clf_resnet.pkl")

# Load flat data
X_flat = np.load("/data_full/resnet_per_frame_X.npy", mmap_mode="r")
y_flat = np.load("/data_full/resnet_per_frame_y.npy", mmap_mode="r")


# X_flat = np.load("/data_full/dino_per_frame_concat/X.npy", mmap_mode="r")   # (total_frames, 2048)
# y_flat = np.load("/data_full/dino_per_frame_concat/y.npy", mmap_mode="r")   # (total_frames,)



# Rebuild videos exactly like training
frames_per_video = compute_frames_per_video(
    "/data_full/real_mini_processed",
    "/data_full/fake_mini_processed"
)

print("Frames in videos:", sum(frames_per_video))
print("Frames in features:", len(X_flat))

X_videos, y_videos = rebuild_videos(X_flat, y_flat, frames_per_video)

train_idx, _, test_idx = load_splits("new_splits.json")

X_train = X_flat[train_idx]
y_train = y_flat[train_idx]



y_true, y_pred, y_proba = evaluate_video_with_metrics(clf, X_videos, y_videos, test_idx)

print("Test idx:", test_idx)
print("Test labels:", np.unique(y_videos[test_idx], return_counts=True))
print("Predicted:", np.unique(y_pred, return_counts=True))
print("clf always predicts?", np.all(y_pred == y_pred[0]))

metrics = compute_metrics(y_true, y_pred, y_proba)
print(metrics)

train_pred = clf.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
print("Training accuracy:", train_acc)

import numpy as np
import os

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

X_flat = np.load("/data_full/resnet_per_frame_X.npy", mmap_mode="r")
y_flat = np.load("/data_full/resnet_per_frame_y.npy", mmap_mode="r")

frames_per_video = compute_frames_per_video(
    "/data_full/real_mini_processed",
    "/data_full/fake_mini_processed"
)

_, y_videos = rebuild_videos(X_flat, y_flat, frames_per_video)

np.save("y_videos.npy", y_videos)
print("Saved y_videos.npy with shape:", y_videos.shape)
print("Label counts:", np.unique(y_videos, return_counts=True))
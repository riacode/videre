import os
import numpy as np
from tqdm import tqdm

patch_dir = "/data_full/resnet_patch_features/"
files = sorted([f for f in os.listdir(patch_dir) if f.endswith("_patch.npy")])

N = len(files)
H = W = 7
D = 2048

# Allocate final arrays
X = np.memmap("resnet_X.npy", dtype=np.float32, mode="w+", shape=(N, D, H, W))
y = np.memmap("resnet_y.npy", dtype=np.int64, mode="w+", shape=(N,))

for i, fname in enumerate(tqdm(files)):
    arr = np.load(os.path.join(patch_dir, fname))   # (49, 2048)

    # reshape into grid
    grid = arr.reshape(H, W, D).transpose(2, 0, 1)  # (2048, 7, 7)

    X[i] = grid

    y[i] = 0 if "academic" in fname else 1

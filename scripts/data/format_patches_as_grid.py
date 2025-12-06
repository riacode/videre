import os
import numpy as np
import argparse
import json
from tqdm import tqdm
from numpy.lib.format import open_memmap

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--grid-h", type=int, default=37)
    parser.add_argument("--grid-w", type=int, default=37)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = [f for f in os.listdir(args.in_dir) if f.endswith(".npy")]

    
    N_samples = len(files)
    H, W = args.grid_h, args.grid_w

    first = np.load(os.path.join(args.in_dir, files[0]))
    N, D = first.shape

    if N != H * W:
        raise ValueError("Grid size mismatch")

    X = open_memmap(
        os.path.join(args.out_dir, "X.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(N_samples, D, H, W)
    )
    y = open_memmap(
        os.path.join(args.out_dir, "y.npy"),
        mode="w+",
        dtype=np.int64,
        shape=(N_samples,)
    )

    for idx, fname in enumerate(tqdm(files, desc="Building dataset")):
        arr = np.load(os.path.join(args.in_dir, fname))  # (H*W, D)
        grid_dhw = arr.reshape(H, W, D).transpose(2, 0, 1)
        X[idx] = grid_dhw
        y[idx] = 0 if "academic" in fname else 1

    X.flush()
    y.flush()

    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(
            {"embedding_dim": D, "grid_h": H, "grid_w": W, "num_patches": H*W},
            f
        )

    print("Done!")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

if __name__ == "__main__":
    main()

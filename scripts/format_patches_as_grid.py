import os
import numpy as np
import argparse
import json
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, required=True, help="Directory containing .npy files")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for grid features")
    parser.add_argument("--grid-h", type=int, default=37, help="Grid height")
    parser.add_argument("--grid-w", type=int, default=37, help="Grid width")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(args.in_dir) if f.endswith(".npy")])

    for fname in tqdm(files, desc="Converting"):
        in_path = os.path.join(args.in_dir, fname)
        out_path = os.path.join(args.out_dir, fname)

        # Load (N, D)
        arr = np.load(in_path)
        if arr.ndim != 2:
            raise ValueError(f"{fname}: expected 2D array (N, D), got shape {arr.shape}")
        N, D = arr.shape
        H, W = args.grid_h, args.grid_w
        if N != H * W:
            raise ValueError(f"{fname}: N={N} does not match H*W={H*W}")

        # Reshape to (H, W, D)
        grid_hw_d = arr.reshape(H, W, D)
        # Transpose to (D, H, W)
        grid_dhw = np.transpose(grid_hw_d, (2, 0, 1))

        # Save
        np.save(out_path, grid_dhw)

    meta = {
        "embedding_dim": D,
        "num_patches": H * W,
        "grid_h": H,
        "grid_w": W
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Conversion done")
    print(f"Saved grid features, meta.json to {args.out_dir}")


if __name__ == "__main__":
    main()

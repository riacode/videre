import os
from tqdm import tqdm
import numpy as np

fake_processed = '/data_full/fake_full_processed/'

fake_files = sorted([f for f in os.listdir(fake_processed) if f.endswith('.npy')])

fake_mini_processed = '/data_full/fake_mini_processed/'


for i, fname in enumerate(fake_files):
    if i % 16 != 0:
        continue
    
    src_path = os.path.join(fake_processed, fname)
    dst_path = os.path.join(fake_mini_processed, fname)

    # skip if already exists
    if os.path.exists(dst_path):
        print(f"Skipping {fname} — already exists.")
        continue

    frames = np.load(src_path)
    print(f"Copying fake video: {fname}, frames={frames.shape}")

    np.save(dst_path, frames)

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as transforms
from typing import Dict
from torch.nn import functional as F
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Identity()     # remove classification head
model.eval()
model.to(device)

# Define the transformation
preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()

# Load and preprocess the image
# Load image from URL

real_processed = '/data_full/real_mini_processed'
fake_processed = '/data_full/fake_mini_processed'


real_files = sorted([f for f in os.listdir(real_processed) if f.endswith('.npy')])
fake_files = sorted([f for f in os.listdir(fake_processed) if f.endswith('.npy')])


real_features = '/data_full/mini_real_resnet_features/'
fake_features = '/data_full/mini_fake_resnet_features/'


@torch.no_grad()
def extract_video_embedding(frames: np.ndarray) -> np.ndarray:
    """Compute mean ResNet50 embedding for all frames in a video."""
    embeddings = []

    for frame in frames:
        image = Image.fromarray(frame.astype(np.uint8))

        img_tensor = preprocess(image).unsqueeze(0).to("cuda")  # (1,3,224,224)

        feat = model(img_tensor)       # (1, 2048)
        feat = feat.squeeze().cpu()    # (2048,)
        embeddings.append(feat)

    stacked = torch.stack(embeddings, dim=0)        # (T, 2048)
    video_embedding = stacked.mean(dim=0)           # (2048,)
    return video_embedding.numpy()


def process_split(files, in_dir, out_dir, label):
    for fname in tqdm(files, desc=label):
        out_file = os.path.join(out_dir, fname.replace('.npy', '_feat.npy'))

        # skip if already processed
        if os.path.exists(out_file):
            print(f"Skipping {fname}, already exists.")
            continue
        print(f"extracting video {fname}")

        frames = np.load(os.path.join(in_dir, fname))
        emb = extract_video_embedding(frames)
        np.save(out_file, emb)


# Run all splits
process_split(real_files, real_processed, real_features, "REAL")
process_split(fake_files,  fake_processed,  fake_features,  "FAKE")

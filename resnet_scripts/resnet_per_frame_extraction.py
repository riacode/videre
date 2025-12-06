import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
import torchvision.transforms.v2 as transforms
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Identity()     
model.eval()
model.to(device)

preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()

real_processed = '/data_full/real_mini_processed'
fake_processed = '/data_full/fake_mini_processed'

real_files = sorted([f for f in os.listdir(real_processed) if f.endswith('.npy')])
fake_files = sorted([f for f in os.listdir(fake_processed) if f.endswith('.npy')])

@torch.no_grad()
def extract_frame_embeddings(frames: np.ndarray) -> np.ndarray:
    """Return a (T, 2048) matrix — one embedding per frame."""
    embeddings = []

    for frame in frames:
        image = Image.fromarray(frame.astype(np.uint8))
        img_tensor = preprocess(image).unsqueeze(0).to(device)  # (1,3,224,224)
        feat = model(img_tensor).squeeze().cpu().numpy()        # (2048,)
        embeddings.append(feat)

    return np.stack(embeddings, axis=0)  # (T, 2048)


X_list = []
y_list = []

print("Extracting REAL embeddings...")
for fname in tqdm(real_files):
    frames = np.load(os.path.join(real_processed, fname))
    feats = extract_frame_embeddings(frames)  # (T, 2048)
    X_list.append(feats)
    y_list.append(np.zeros(feats.shape[0], dtype=np.int32))  # label = 0

print("Extracting FAKE embeddings...")
for fname in tqdm(fake_files):
    frames = np.load(os.path.join(fake_processed, fname))
    feats = extract_frame_embeddings(frames)  # (T, 2048)
    X_list.append(feats)
    y_list.append(np.ones(feats.shape[0], dtype=np.int32))  # label = 1

X_all = np.vstack(X_list)       # (total_frames, 2048)
y_all = np.concatenate(y_list)  # (total_frames,)

print("Final shapes:")
print("X_all:", X_all.shape)
print("y_all:", y_all.shape)

np.save("/data_full/resnet_per_frame_X.npy", X_all)
np.save("/data_full/resnet_per_frame_y.npy", y_all)

print("Saved combined frame embeddings.")

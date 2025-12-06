import torch
import numpy as np
import timm
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True)
model.eval().to(device)

# DINOv2 embedding dimension for this model = 384
EMB_DIM = model.num_features  # should be 384


transform = transforms.Compose([
    transforms.Resize(518),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


real_processed = '/data_full/real_mini_processed'
fake_processed = '/data_full/fake_mini_processed'

real_files = sorted([f for f in os.listdir(real_processed) if f.endswith('.npy')])
fake_files = sorted([f for f in os.listdir(fake_processed) if f.endswith('.npy')])


@torch.no_grad()
def extract_frame_embeddings(frames: np.ndarray) -> np.ndarray:
    """
    Returns a (T, 384) matrix — one DINOv2 embedding per frame.
    """
    embeddings = []

    for frame in frames:
        pil = Image.fromarray(frame.astype(np.uint8))
        img = transform(pil).unsqueeze(0).to(device)     # (1, 3, H, W)

        emb = model(img)                                 # (1, 384)
        emb = emb.squeeze(0).cpu().numpy()               # (384,)

        embeddings.append(emb)

    return np.stack(embeddings, axis=0)  # (T, 384)

X_list = []
y_list = []

print("Extracting REAL embeddings...")
for fname in tqdm(real_files):
    frames = np.load(os.path.join(real_processed, fname))
    feats = extract_frame_embeddings(frames)       # (T, 384)
    X_list.append(feats)
    y_list.append(np.zeros(feats.shape[0], dtype=np.int32))  # label = 0

print("Extracting FAKE embeddings...")
for fname in tqdm(fake_files):
    frames = np.load(os.path.join(fake_processed, fname))
    feats = extract_frame_embeddings(frames)       # (T, 384)
    X_list.append(feats)
    y_list.append(np.ones(feats.shape[0], dtype=np.int32))  # label = 1


X_all = np.vstack(X_list)       # (total_frames, 384)
y_all = np.concatenate(y_list)  # (total_frames,)

print("Final shapes:")
print("X_all:", X_all.shape)    # should be (#frames, 384)
print("y_all:", y_all.shape)

np.save("/home/irisxu/videre/features/dino_tree/dino_per_frame_X.npy", X_all)
np.save("/home/irisxu/videre/features/dino_tree/dino_per_frame_y.npy", y_all)

print("Saved per-frame DINOv2 embeddings.")

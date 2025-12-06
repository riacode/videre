import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Identity()  # remove classifier
model.eval()
model.to(device)

feature_extractor = nn.Sequential(*list(model.children())[:-2]).to(device)
# shape: (B, 2048, H_feat, W_feat)

preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()

@torch.no_grad()
def extract_resnet_patch_tokens(frames_np, batch_size=8):
    """Return (T, N_patches, 2048) where N_patches = H_feat * W_feat."""
    all_tokens = []

    for i in range(0, len(frames_np), batch_size):
        batch = frames_np[i:i+batch_size]

        imgs = [preprocess(Image.fromarray(f.astype(np.uint8))) for f in batch]
        x = torch.stack(imgs).to(device)

        feats = feature_extractor(x)  # (B, 2048, Hf, Wf)
        B, C, Hf, Wf = feats.shape

        tokens = feats.reshape(B, C, Hf*Wf).permute(0, 2, 1)  # (B, N_patches, 2048)
        all_tokens.append(tokens.cpu())

    return torch.cat(all_tokens, dim=0)  # (T, N_patches, 2048)

def extract_video_patch_grid(frames_np, reduce="mean"):
    tokens = extract_resnet_patch_tokens(frames_np)  # (T, N, 2048)

    if reduce == "mean":
        return tokens.mean(0).numpy()          # (N, 2048)
    elif reduce == "median":
        return tokens.median(0).values.numpy() # (N, 2048)
    else:
        raise ValueError("reduce must be mean or median")


def process_folder(real_dir, fake_dir, out_dir, device="cuda"):
    os.makedirs(out_dir, exist_ok=True)

    for label_dir, tag in [(real_dir, "REAL"), (fake_dir, "FAKE")]:
        files = sorted([f for f in os.listdir(label_dir) if f.endswith(".npy")])

        for fname in tqdm(files, desc=tag):
            out_file = os.path.join(out_dir, fname.replace(".npy", "_patch.npy"))

            # Skip already processed files
            if os.path.exists(out_file):
                print(f"Skipping {fname}, already exists.")
                continue

            frames = np.load(os.path.join(label_dir, fname))   # (T, H, W, C)
            patches = extract_video_patch_grid(frames, reduce="mean")
            np.save(out_file, patches)

            del frames, patches
            if device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    real_dir = "/data_full/real_full_processed/"
    fake_dir = "/data_full/fake_full_processed/"
    out_dir = "/data_full/resnet_patch_features/"

    process_folder(real_dir, fake_dir, out_dir, device="cuda")

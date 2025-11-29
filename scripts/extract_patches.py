import os
import numpy as np
from PIL import Image
import torch
import timm
import torchvision.transforms.v2 as transforms
from tqdm import tqdm
from torch.amp import autocast
import json
from torchvision.transforms.v2 import Compose, Resize, CenterCrop, ToImage, ToDtype, Normalize
import torch

timm.layers.set_fused_attn(False)

def load_model(device):
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True)
    model.eval()
    model.to(device)
    return model


device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(device)


transform = Compose([
    Resize(518),
    CenterCrop(518),
    ToImage(),
    ToDtype(torch.float32, scale=True),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def extract_patch_tokens_single_batch(frames_np, device="cuda"):
    imgs = [transform(Image.fromarray(f.astype(np.uint8))) for f in frames_np]
    x = torch.stack(imgs, 0).to(device, non_blocking=True)

    tokens = {}

    def hook_fn(module, input, output):
        tokens["x"] = output  # (B, 1+N, 384)

    handle = model.blocks[-1].register_forward_hook(hook_fn)
    with autocast("cuda", dtype=torch.float16):  # experimental optimization idk man
        _ = model(x)
    handle.remove()

    out = tokens["x"]              # (B, 1+N, 384)
    patches = out[:, 1:, :]        # drop CLS → (B, N, 384)

    del x
    return patches.cpu()


def extract_patch_tokens_batch(frames_np, batch_size=8, device="cuda"):
    all_patches = []

    for i in range(0, len(frames_np), batch_size):
        batch = frames_np[i:i + batch_size]
        patches = extract_patch_tokens_single_batch(batch, device)
        all_patches.append(patches)

    return torch.cat(all_patches, dim=0)  # (T, N, 384)


def extract_video_patch_grid_batch(frames, device="cuda", reduce="mean"):
    tokens = extract_patch_tokens_batch(frames, device=device)

    if reduce == "mean":
        return tokens.mean(0).numpy()
    elif reduce == "median":
        return tokens.median(0).values.numpy()
    else:
        raise ValueError("reduce must be mean or median")


def process_folder(real_dir, fake_dir, out_dir, device="cuda"):
    model.to(device)
    os.makedirs(out_dir, exist_ok=True)

    for label_dir, tag in [(real_dir, "REAL"), (fake_dir, "FAKE")]:
        files = sorted([f for f in os.listdir(label_dir) if f.endswith(".npy")])

        for fname in tqdm(files, desc=tag):
            out_file = os.path.join(out_dir, fname.replace(".npy", "_patch.npy"))
            if os.path.exists(out_file):
                continue

            frames = np.load(os.path.join(label_dir, fname))
            patches = extract_video_patch_grid_batch(frames, device=device)
            np.save(out_file, patches)

            del frames, patches
            if device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    real_dir = "/data_full/real_full_processed/"
    fake_dir = "/data_full/fake_full_processed/"
    out_dir = "/data_full/dino_patch_features/"

    process_folder(real_dir, fake_dir, out_dir, device="cuda")

import numpy as np
import cv2
import json
import os
import argparse
import logging
import torch
import torch.nn.functional as F

from videre.models.torch_models import PatchCNN, GradCAM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one video using GradCAM.")
    parser.add_argument("--video-name", type=str, required=True, help="Name of the input .mp4 video")
    parser.add_argument("--grid-feature-dir", type=str, required=True, help="Directory containing X.npy, y.npy, meta.json")
    parser.add_argument("--og-feature-dir", type=str, required=True, help="Directory containing X.npy, y.npy, meta.json")
    parser.add_argument("--run-name", type=str, required=True, help="Folder name of the trained run")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory containing the run outputs")
    parser.add_argument("--output-video", type=str, default="cam_overlay.mp4")
    parser.add_argument("--alpha", type=float, default=0.4)
    return parser.parse_args()

def preprocess_video_frame_for_cam(frame):
    h, w = frame.shape[:2]

    if h < w:
        new_h = 518
        new_w = int(w * 518 / h)
    else:
        new_w = 518
        new_h = int(h * 518 / w)

    frame_resized = cv2.resize(frame, (new_w, new_h))

    x1 = (new_w - 518) // 2
    y1 = (new_h - 518) // 2
    cropped = frame_resized[y1:y1+518, x1:x1+518]

    return cropped

def overlay_heatmap_on_video(video_path, heatmap, output_path, alpha=0.4):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    
    heatmap_resized = cv2.resize(heatmap, (518, 518))
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (518, 518))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped = preprocess_video_frame_for_cam(frame)
        blended = cv2.addWeighted(heatmap_color, alpha, cropped, 1 - alpha, 0)
        out.write(blended)

    cap.release()
    out.release()
    logger.info(f"Saved heatmap overlay video to: {output_path}")


def load_feature_tensor(og_feature_dir, grid_feature_dir, video_name, D, H, W, device):
    base = os.path.splitext(os.path.basename(video_name))[0]
    print(base)

    feature_files = sorted([f for f in os.listdir(og_feature_dir) if f.endswith("_patch.npy")])

    match = None
    for i, f in enumerate(feature_files):
        if base in f:
            match = i
            break

    if match is None:
        raise ValueError(f"Could not find patch feature for video: {video_name}")

    logger.info(f"Using feature index {match} from X.npy")
    
    X_path = os.path.join(grid_feature_dir, "X.npy")
    X_mm = np.load(X_path, mmap_mode="r")  # shape (N, D, H, W)

    arr = X_mm[match]  # (D, H, W)

    x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)  # (1, D, H, W)
    return x


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    with open(os.path.join(args.grid_feature_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    D = meta["embedding_dim"]
    H = meta["grid_h"]
    W = meta["grid_w"]

    model_path = os.path.join(args.output_dir, "models", f"{args.run_name}.pt")
    logger.info(f"Loading model from: {model_path}")

    model = PatchCNN(embedding_dim=D, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    target_layer = model.conv3
    cam = GradCAM(model, target_layer)
    x = load_feature_tensor(args.og_feature_dir, args.grid_feature_dir, args.video_name, D, H, W, device)
    x.requires_grad_(True)

    cam_map = cam(x)  # expected shape: (1, 1, H, W) or (H, W)

    if cam_map.ndim == 4:
        cam_map = cam_map[0, 0]
    elif cam_map.ndim == 3:
        cam_map = cam_map[0]

    cam_map = F.relu(cam_map)
    cam_map = cam_map / (cam_map.max() + 1e-6)
    cam_np = cam_map.cpu().numpy()

    # Show top 5 most important patches
    flat = cam_np.flatten()
    topk_idx = flat.argsort()[-5:][::-1]
    topk_scores = flat[topk_idx]
    logger.info(f"Top patches: {topk_idx}")
    logger.info(f"Scores: {topk_scores}")
    
    overlay_heatmap_on_video(
        video_path=args.video_name,
        heatmap=cam_np,
        output_path=args.output_video,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()

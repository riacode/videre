
import numpy as np
import cv2
import json
import os
import argparse
import logging
import random
import torch
import torch.nn.functional as F

from videre.models.torch_models import PatchCNN, GradCAM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="CAM evaluation with bbox accuracy checking")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video-name", type=str,
                       help="Single MP4 video path to evaluate")
    group.add_argument("--video-dir", type=str,
                       help="Directory containing multiple MP4 videos")

    parser.add_argument("--num-random", type=int, default=0,
                        help="Number of random videos to evaluate from --video-dir")

    parser.add_argument("--json-path", type=str, required=True,
                        help="JSON metadata file containing bbox information")

    parser.add_argument("--grid-feature-dir", type=str, required=True)
    parser.add_argument("--og-feature-dir", type=str, required=True)

    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--output-video", type=str, default="cam_overlay.mp4")
    parser.add_argument("--boxes-video", type=str, default="cam_overlay_boxes.mp4")
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
    resized = cv2.resize(frame, (new_w, new_h))
    x1 = (new_w - 518) // 2
    y1 = (new_h - 518) // 2
    return resized[y1:y1+518, x1:x1+518]

def map_bbox_to_cropped_frame(video_path, bbox):
    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if orig_h < orig_w:
        new_h = 518
        new_w = int(orig_w * 518 / orig_h)
    else:
        new_w = 518
        new_h = int(orig_h * 518 / orig_w)

    x1 = (new_w - 518) // 2
    y1 = (new_h - 518) // 2

    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    return {
        "left": int(bbox["left"] * scale_x) - x1,
        "top": int(bbox["top"] * scale_y) - y1,
        "width": int(bbox["width"] * scale_x),
        "height": int(bbox["height"] * scale_y),
    }

def overlay_heatmap_on_video(video_path, heatmap, output_path, alpha=0.4):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    heatmap_resized = cv2.resize(heatmap, (518, 518))
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (518, 518))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped = preprocess_video_frame_for_cam(frame)
        blended = cv2.addWeighted(heatmap_color, alpha, cropped, 1 - alpha, 0)
        writer.write(blended)

    cap.release()
    writer.release()
    logger.info(f"Saved CAM overlay: {output_path}")

def load_feature_tensor(og_feature_dir, grid_feature_dir, video_name, D, H, W, device):
    base = os.path.splitext(os.path.basename(video_name))[0]
    feature_files = sorted(
        [f for f in os.listdir(og_feature_dir) if f.endswith("_patch.npy")]
    )

    matched_idx = None
    for i, fname in enumerate(feature_files):
        if base in fname:
            matched_idx = i
            break

    if matched_idx is None:
        raise ValueError(f"No patch feature found for {video_name}")

    X_path = os.path.join(grid_feature_dir, "X.npy")
    X_mm = np.load(X_path, mmap_mode="r")  # shape: (N, D, H, W)
    arr = X_mm[matched_idx]

    x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)
    return x

def patch_index_to_center(idx, H, W, frame_px=518):
    row = idx // W
    col = idx % W

    patch_h = frame_px / H
    patch_w = frame_px / W

    cy = row * patch_h + patch_h / 2
    cx = col * patch_w + patch_w / 2

    return int(cx), int(cy)

def is_inside_bbox(x, y, bbox):
    left = bbox["left"]
    top = bbox["top"]
    width = bbox["width"]
    height = bbox["height"]

    return (left <= x <= left + width) and (top <= y <= top + height)

def overlay_bbox_and_patches(video_path, bbox_resized, patch_centers, output_path, alpha=0.4):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (518, 518))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop/resize to 518x518
        frame_cropped = preprocess_video_frame_for_cam(frame)

        left = bbox_resized["left"]
        top = bbox_resized["top"]
        width = bbox_resized["width"]
        height = bbox_resized["height"]
        cv2.rectangle(frame_cropped, (left, top), (left + width, top + height), (0, 255, 0), 2)

        for cx, cy in patch_centers:
            cv2.circle(frame_cropped, (cx, cy), radius=5, color=(0, 0, 255), thickness=-1)

        writer.write(frame_cropped)

    cap.release()
    writer.release()
    logger.info(f"Saved video with bbox and patch overlay: {output_path}")


def evaluate_video(video_name, cam_np, video_to_entry, H, W):
    base = os.path.basename(video_name)
    entry = video_to_entry.get(base, None)

    if entry is None:
        raise ValueError(f"No JSON metadata entry for {video_name}")

    bbox = entry["faketrace_bbox"]

    # Top patches
    flat = cam_np.flatten()
    topk = flat.argsort()[-20:][::-1]

    centers = [patch_index_to_center(idx, H, W) for idx in topk]

    resized = map_bbox_to_cropped_frame(video_name, bbox)

    hit = any(is_inside_bbox(cx, cy, resized) for cx, cy in centers)

    return int(hit), centers, topk

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # grid meta
    with open(os.path.join(args.grid_feature_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    D = meta["embedding_dim"]
    H = meta["grid_h"]
    W = meta["grid_w"]

    with open(args.json_path, "r") as f:
        json_list = json.load(f)

    # Build fast lookup table
    video_to_entry = {
        os.path.basename(item["video_path"]): item
        for item in json_list
    }

    # Load model
    model_path = os.path.join(args.output_dir, "models", f"{args.run_name}.pt")
    logger.info(f"Loading model from {model_path}")

    model = PatchCNN(embedding_dim=D, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    target_layer = model.conv3
    cam_extractor = GradCAM(model, target_layer)

    # single video
    if args.video_name is not None:
        video = args.video_name

        x = load_feature_tensor(args.og_feature_dir, args.grid_feature_dir,
                                video, D, H, W, device)
        x.requires_grad_(True)

        cam_map = cam_extractor(x)
        if cam_map.ndim == 4:
            cam_map = cam_map[0, 0]
        elif cam_map.ndim == 3:
            cam_map = cam_map[0]

        cam_map = F.relu(cam_map)
        cam_map = cam_map / (cam_map.max() + 1e-6)
        cam_np = cam_map.cpu().numpy()

        acc, centers, topk = evaluate_video(video, cam_np, video_to_entry, H, W)

        logger.info(f"Top patch indices: {topk}")
        logger.info(f"Centers: {centers}")
        logger.info(f"Accuracy for this video = {acc}")

        overlay_heatmap_on_video(
            video_path=video,
            heatmap=cam_np,
            output_path=args.output_video,
            alpha=args.alpha,
        )

        base = os.path.basename(video)
        entry = video_to_entry.get(base, None)
    
        if entry is None:
            raise ValueError(f"No JSON metadata entry for {video_name}")
    
        bbox = entry["faketrace_bbox"]
        resized = map_bbox_to_cropped_frame(video, bbox)
        overlay_bbox_and_patches(
            video_path=video,
            bbox_resized=resized,
            patch_centers=centers,
            output_path=args.boxes_video
        )

        return

    # Multi-video mode
    else:
        video_dir = args.video_dir
        num = args.num_random

        all_videos = [
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.lower().endswith(".mp4")
        ]

        if num > 0:
            videos = random.sample(all_videos, num)
        else:
            videos = all_videos

        total = 0
        correct = 0

        for vid in videos:
            total += 1

            x = load_feature_tensor(args.og_feature_dir, args.grid_feature_dir,
                                    vid, D, H, W, device)
            x.requires_grad_(True)

            cam_map = cam_extractor(x)
            if cam_map.ndim == 4:
                cam_map = cam_map[0, 0]
            elif cam_map.ndim == 3:
                cam_map = cam_map[0]

            cam_map = F.relu(cam_map)
            cam_map = cam_map / (cam_map.max() + 1e-6)
            cam_np = cam_map.cpu().numpy()

            acc, _, _ = evaluate_video(vid, cam_np, video_to_entry, H, W)
            correct += acc

        logger.info(f"Evaluated {total} videos")
        logger.info(f"Accuracy = {correct / total:.4f}")


if __name__ == "__main__":
    main()
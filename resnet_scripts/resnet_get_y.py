
def process_folder(real_dir, fake_dir, out_dir, device="cuda"):
    os.makedirs(out_dir, exist_ok=True)

    for label_dir, tag in [(real_dir, "REAL"), (fake_dir, "FAKE")]:
        files = sorted([f for f in os.listdir(label_dir) if f.endswith(".npy")])

        for fname in tqdm(files, desc=tag):
            out_file = os.path.join(out_dir, fname.replace(".npy", "_patch.npy"))

            frames = np.load(os.path.join(label_dir, fname))   # (T, H, W, C)
            patches = extract_video_patch_grid(frames, reduce="mean")
            np.save(out_file, patches)

            del frames, patches
            if device == "cuda":
                torch.cuda.empty_cache()


    real_files = sorted([f for f in os.listdir(real_) if f.endswith('.npy')])

    for i, fname in enumerate()

if __name__ == "__main__":
    real_dir = "/data_full/real_full_processed/"
    fake_dir = "/data_full/fake_full_processed/"
    out_dir = "/data_full/resnet_y.py"

    process_folder()
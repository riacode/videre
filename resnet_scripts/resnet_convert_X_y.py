import os
import numpy as np

def load_and_stack(dir1, dir2, output_file="combined.npy"):
    arrays = []

    for d in [dir1, dir2]:
        for fname in sorted(os.listdir(d)):
            if fname.endswith(".npy"):
                path = os.path.join(d, fname)
                print("Loading:", path)
                arr = np.load(path)
                arrays.append(arr)

    stacked = np.stack(arrays, axis=0)
    print("Final shape:", stacked.shape)
    np.save(output_file, stacked)
    print("Saved to:", output_file)



if __name__ == "__main__":
    real_dir = "/data_full/mini_real_resnet_features/"
    fake_dir = "/data_full/mini_fake_resnet_features/"
    out_dir = "/data_full/resnet_X.npy"

    load_and_stack(real_dir, fake_dir, out_dir)

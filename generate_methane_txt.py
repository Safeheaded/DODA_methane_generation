import os
import numpy as np
from sklearn.model_selection import train_test_split


def load_directories_to_numpy(path):
    # List all items in the path and filter only directories
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return np.array(dirs)


if __name__ == "__main__":
    # Update with your desired directory path
    directory_path = "./datasets/methane/data"
    directories_array = load_directories_to_numpy(directory_path)
    X_train, X_test, _, _ = train_test_split(
        directories_array, directories_array, test_size=0.33, random_state=42
    )
    np.savetxt("./datasets/methane/train_cldm.txt", X_train, fmt="%s")
    np.savetxt("./datasets/methane/train_ldm.txt", X_train, fmt="%s")

    np.savetxt("./datasets/methane/val_cldm.txt", X_train, fmt="%s")
    np.savetxt("./datasets/methane/val_ldm.txt", X_train, fmt="%s")

import os
import shutil
import random
from os import path, walk

DATA_DIR = "data"
DATASET_DIR = path.join(DATA_DIR, "brain_tumor_mri")
TRAIN_DIR = "brain_tumor_mri_train"
TEST_DIR = "brain_tumor_mri_test"


def main():
    for _dir in os.listdir(DATASET_DIR):
        _path = path.join(DATASET_DIR, _dir)
        if path.isdir(_path):
            train_path = path.join(DATA_DIR, TRAIN_DIR, _dir)
            test_path = path.join(DATA_DIR, TEST_DIR, _dir)
            if not path.lexists(train_path):
                print("creating path:", train_path)
                os.makedirs(train_path)

            if not path.lexists(test_path):
                print("creating path:", test_path)
                os.makedirs(test_path)

            for root, dirs, files in walk(_path):
                for f in files:
                    copy_dir = TRAIN_DIR if random.random() < 0.9 else TEST_DIR
                    shutil.copyfile(
                        path.join(root, f), path.join(DATA_DIR, copy_dir, _dir, f)
                    )


if __name__ == "__main__":
    main()

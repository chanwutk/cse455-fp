import base64
import random
from os import listdir, path


def image_to_base64(filename: str):
    with open(filename, "rb") as img_file:
        data = base64.b64encode(img_file.read())
    return data.decode("utf-8")


def random_images_to_base64(directory: str, num: int):
    def join_dir(filename: str):
        return path.join(directory, filename)

    dirs = map(join_dir, listdir(directory))
    images = list(filter(path.isfile, dirs))
    random.shuffle(images)
    return list(map(image_to_base64, images[:num]))

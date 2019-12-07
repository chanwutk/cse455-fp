import base64
import random
from os import listdir, path


def image_to_base64(filename: str):
    with open(filename, "rb") as img_file:
        data = base64.b64encode(img_file.read())
    return data.decode("utf-8")


def random_images_to_base64(directory: str, num: int):
    dirs = listdir(directory)

    def is_file(d):
        path.isfile(path.join(directory, d))

    images = filter(is_file, dirs)
    random.shuffle(images)
    return list(map(image_to_base64, images[:num]))

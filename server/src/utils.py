import base64
import random
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from os import listdir, path
from torchvision import transforms
from PIL import Image
from io import BytesIO


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


def pil_to_base64(pil_image):
    byte_buffer = io.BytesIO()
    pil_image.save(byte_buffer, format="JPEG")

    # reset file pointer to start
    byte_buffer.seek(0)
    img_bytes = byte_buffer.read()

    return base64.b64encode(img_bytes).decode("ascii")


def normalize_tensor(tensor):
    tensor = tensor.float()
    min_value = torch.min(tensor).item()
    max_value = torch.max(tensor).item()
    range_value = max_value - min_value
    if range_value > 0:
        return (tensor - min_value) / range_value
    else:
        return torch.zeros(tensor.size())


def tensor_to_image(tensor):
    normalized = normalize_tensor(tensor)
    return TF.to_pil_image(normalized)


def mask_input_with_output(image: torch.Tensor, output: torch.Tensor):
    output = normalize_tensor(output)
    output = nn.UpsamplingNearest2d(size=(224, 224))(output)
    # ones = torch.zeros_like(output)
    # output = torch.cat([ones, output, ones], 1)
    # output = ones - output
    return pil_to_base64(TF.to_pil_image((image * output).squeeze(dim=0)))


def mask_input_with_outputs(image, trace):
    output = []
    # bound = trace[1].size()[1]
    # channel = random.sample(range(0, bound), num)
    # for c in channel:
    for c in range(trace[1].size()[1]):
        masked = mask_input_with_output(image, trace[:, c : c + 1, :, :])
        output.append(masked)
    return output

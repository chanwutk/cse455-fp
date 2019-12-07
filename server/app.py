import os
import re

from src.networks import PneumoniaVGG
from src.data_loaders import load_base64_image, load_classes
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from src.utils import random_images_to_base64

app = Flask(__name__)
app.config["CORS_HEADERS"] = "Content-Type"
cors = CORS(
    app,
    resources={r"/classify": {"origins": "*"}, r"/example-images": {"origins": "*"}},
)

cwd = os.getcwd()
weights_path = os.path.join(cwd, "data/weights.pt")
dataset_dir = os.path.join(cwd, "data/chest_xray")

model = PneumoniaVGG()
model.load_model(weights_path)
class_names = load_classes("class_names.txt")


@app.route("/classify", methods=["POST"])
@cross_origin(origin="localhost", headers=["Content-Type"])
def classify():
    if request.method != "POST":
        return "/classify only accept POST request", 500

    data = re.sub("^data:image/.+;base64,", "", request.json["data"])
    image = load_base64_image(data, do_normalize=True)
    output = model.classify(image)
    class_idx = output.view(-1).max(0)[1]
    return class_names[class_idx.item()]


@app.route("/example-images/<is_normal>/<n>", methods=["GET"])
@cross_origin(origin="localhost", headers=["Content-Type"])
def example_images(is_normal: str, n: str):
    example_dir = "train/" + ("NORMAL" if is_normal == "true" else "PNEUMONIA")
    output = random_images_to_base64(os.path.join(dataset_dir, example_dir), int(n))
    print(output)
    return jsonify(output)


if __name__ == "__main__":
    app.run(port=6543)

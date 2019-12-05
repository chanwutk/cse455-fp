import os
import re

from src.networks import PneumoniaVGG
from src.data_loaders import load_base64_image, load_classes
from flask import Flask, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
app.config["CORS_HEADERS"] = "Content-Type"
cors = CORS(app, resources={r"/classify": {"origins": "*"}})

cwd = os.getcwd()
weights_path = os.path.join(cwd, "data/weights.pt")

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


if __name__ == "__main__":
    app.run(port=6543)

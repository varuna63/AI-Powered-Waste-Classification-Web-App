import os
from flask import Flask, request, render_template, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import threading
import webbrowser

# ----------------- CONFIG -----------------
UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "efficientnet_waste.pth"
CLASS_NAMES = {
    1: "glass",
    2: "battery",
    3: "biological",
    4: "clothes",
    5: "metal",
    6: "paper",
    7: "plastic",
    8: "trash"
}

# ----------------- MODEL -----------------
num_classes = 8
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = val_transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = outputs.max(1)
    return CLASS_NAMES[predicted.item() + 1]

# ----------------- FLASK APP -----------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)
    prediction = predict_image(filepath)
    return jsonify({"filename": file.filename, "prediction": prediction})

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    files = request.files.getlist("files")
    results = []
    for file in files:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        prediction = predict_image(filepath)
        results.append({"filename": file.filename, "prediction": prediction})
    return jsonify(results)

# ----------------- AUTO OPEN BROWSER -----------------
def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    threading.Timer(1, open_browser).start()
    app.run(host="127.0.0.1", port=5000, debug=True)

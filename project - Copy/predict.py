import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template
import webbrowser

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Class labels
class_labels = {
    1: "glass",
    2: "battery",
    3: "biological",
    4: "clothes",
    5: "metal",
    6: "paper",
    7: "plastic",
    8: "trash"
}

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
model.load_state_dict(torch.load("efficientnet_waste.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_t = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)

    return class_labels[predicted.item() + 1]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"})

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    prediction = predict_image(filepath)

    return jsonify({
        "filename": file.filename,
        "prediction": prediction,
        "image_url": f"/static/uploads/{file.filename}"
    })

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"})

    results = []
    for file in files:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        prediction = predict_image(filepath)
        results.append({
            "filename": file.filename,
            "prediction": prediction,
            "image_url": f"/static/uploads/{file.filename}"
        })

    return jsonify(results)

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from flask import Flask, request, render_template_string
from PIL import Image
import io

app = Flask(__name__)

# Load Model
MODEL_PATH = 'batik_type_model.pth'

# Define model architecture (sesuaikan dengan arsitektur model Anda)
class BatikClassifier(nn.Module):
    def __init__(self):
        super(BatikClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BatikClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("✅ System Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading system: {e}")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_bytes):
    # Decode gambar
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Apply transformations
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    return img_tensor

@app.route('/', methods=['GET'])
def home():
    return render_template_string('''
    <div style="text-align:center; padding:50px;">
        <h1>Klasifikasi Jenis Batik</h1>
        <p>Upload gambar batik untuk mengklasifikasi jenis batik:</p>
        <p><b>Batik Cap</b> atau <b>Batik Tulis</b></p>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required><br><br>
            <button type="submit" style="padding:10px 20px; font-size:16px;">Prediksi</button>
        </form>
    </div>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        img_tensor = preprocess_image(file.read())
        
        # Prediction
        with torch.no_grad():
            img_tensor = img_tensor.to(device)
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Class labels: 0 = Batik Tulis, 1 = Batik Cap
        class_names = ['Batik Tulis', 'Batik Cap']
        predicted_class = class_names[predicted.item()]
        
        return render_template_string('''
        <div style="text-align:center; padding:50px;">
            <h1>Hasil Klasifikasi</h1>
            <h2 style="color:#2E7D32;">{{ predicted_class }}</h2>
            <br>
            <a href="/" style="padding:10px 20px; background:#1976D2; color:white; text-decoration:none; border-radius:5px;">Kembali</a>
        </div>
        ''', predicted_class=predicted_class)
    except Exception as e:
        return f"<h2 style='text-align:center; color:red;'>Error: {e}</h2><center><a href='/'>Kembali</a></center>"

if __name__ == '__main__':
    # Port 7860 wajib untuk Hugging Face Spaces
    app.run(host='0.0.0.0', port=7860)
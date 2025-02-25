import platform
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile
from torchvision import transforms
from PIL import Image
import io
import os

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "/mnt/models/mnist_cnn.pt")

from train import Net  # Reuse the same Net definition from train.py
model = Net()

# Determine device: use CUDA if available; on macOS, try MPS if available; otherwise, fallback to CPU.
if torch.cuda.is_available():
    device = torch.device("cuda")
elif platform.system().lower() == "darwin" and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Load model weights and move the model to the chosen device
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.post("/predict")
async def predict(file: UploadFile):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    # Preprocess and move the image to the chosen device
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    pred = output.argmax(dim=1).item()
    return {"prediction": pred}

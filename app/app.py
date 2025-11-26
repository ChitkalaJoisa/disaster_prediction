from fastapi import FastAPI, UploadFile, File, Form
import torch
import tensorflow as tf
import numpy as np
from PIL import Image

app = FastAPI()

# -------------------------------
# Load PyTorch models (images)
# -------------------------------
deeplab_model = torch.load("models/best_deeplabv3.pth", map_location="cpu")
landslide_seg_model = torch.load("models/landslide_segmentation.pth", map_location="cpu")

deeplab_model.eval()
landslide_seg_model.eval()

# -------------------------------
# Load Keras models (weather)
# -------------------------------
flood_weather_model = tf.keras.models.load_model("models/flood_risk_regression_model.h5")
landslide_lstm_model = tf.keras.models.load_model("models/landslide_lstm_model.h5")

# -------------------------------
# Helpers
# -------------------------------
def process_image(file):
    img = Image.open(file).convert("RGB")
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0)
    return img

def future_weather_input(temp, humid, rain, wind):
    return np.array([[temp, humid, rain, wind]])

# -------------------------------
# FLOOD PREDICTION
# -------------------------------
@app.post("/predict/flood")
async def predict_flood(
    file: UploadFile = File(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    rainfall: float = Form(...),
    windspeed: float = Form(...)
):

    # IMAGE → segmentation
    img_tensor = process_image(file.file)
    with torch.no_grad():
        flood_mask = deeplab_model(img_tensor)

    # Convert to binary prediction (1 yes, 0 no)
    image_pred = 1 if flood_mask.sum() > 100 else 0

    # WEATHER MODEL → percentage
    weather_input = future_weather_input(temperature, humidity, rainfall, windspeed)
    percent = float(flood_weather_model.predict(weather_input)[0][0] * 100)

    return {
        "image_prediction": image_pred,
        "future_prediction": round(percent, 2)
    }


# -------------------------------
# LANDSLIDE PREDICTION
# -------------------------------
@app.post("/predict/landslide")
async def predict_landslide(
    file: UploadFile = File(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    rainfall: float = Form(...),
    windspeed: float = Form(...)
):
    # IMAGE → segmentation
    img_tensor = process_image(file.file)
    with torch.no_grad():
        landslide_mask = landslide_seg_model(img_tensor)

    image_pred = 1 if landslide_mask.sum() > 100 else 0

    # WEATHER → LSTM model
    weather_input = future_weather_input(temperature, humidity, rainfall, windspeed)
    percent = float(landslide_lstm_model.predict(weather_input)[0][0] * 100)

    return {
        "image_prediction": image_pred,
        "future_prediction": round(percent, 2)
    }

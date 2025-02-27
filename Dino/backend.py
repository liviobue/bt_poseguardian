import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import datetime
import timm  # DINO ist in timm verfügbar

# Initialisiere FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Connection
MONGO_URI = "mongodb+srv://livio:Zuerich578@cluster0.axdzry5.mongodb.net/test"
client = MongoClient(MONGO_URI)
db = client["bt_poseguardian"]
collection = db["data"]

# Lade das DINO Model
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
model.eval()

# OpenCV Webcam
cap = cv2.VideoCapture(0)

# Transformation für das Modell
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.get("/keypoints")
async def get_keypoints():
    ret, frame = cap.read()
    if not ret:
        return JSONResponse(content={"error": "Unable to access webcam"}, status_code=500)

    # OpenCV Bild in PIL umwandeln
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)

    # Extrahiere Merkmale aus dem letzten Layer (du kannst es optimieren)
    keypoints = output.squeeze().tolist()[:10]  # Hier nehme ich einfach die ersten 10 Werte als Beispiel

    # Speichere Keypoints in MongoDB
    document = {
        "timestamp": datetime.datetime.utcnow(),
        "keypoints": keypoints
    }
    #collection.insert_one(document)

    return {"message": "Keypoints saved", "keypoints": keypoints}


@app.get("/video_feed")
async def video_feed():
    def generate_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Encode frame als JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.on_event("shutdown")
def shutdown_event():
    cap.release()
    client.close()  # MongoDB schließen


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image
from pymongo import MongoClient
import datetime
from transformers import ViTModel, AutoImageProcessor

# Load DINO Model
model_name = "facebook/dino-vits16"
processor = AutoImageProcessor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name)  # Use ViTModel instead of DINOModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["http://localhost:3000"] for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Connection
MONGO_URI = "mongodb+srv://livio:Zuerich578@cluster0.axdzry5.mongodb.net/test"
client = MongoClient(MONGO_URI)
db = client["bt_poseguardian"]
collection = db["data"]

cap = cv2.VideoCapture(0)  # Open webcam

def extract_features(frame):
    """Extracts features from a video frame using DINO."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    features = outputs.last_hidden_state.squeeze(0).cpu().numpy()
    return features.tolist()

@app.get("/keypoints")
async def get_keypoints():
    ret, frame = cap.read()
    if not ret:
        raise HTTPException(status_code=500, detail="Unable to access webcam")

    keypoints = extract_features(frame)
    
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
            
            keypoints = extract_features(frame)
            for point in keypoints[:10]:  # Visualizing some keypoints
                x, y = int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.on_event("shutdown")
def shutdown_event():
    cap.release()
    client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

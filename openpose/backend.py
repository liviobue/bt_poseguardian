from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import cv2
import datetime
import os
import numpy as np

# OpenPose Setup
OPENPOSE_PATH = "/openpose"
os.environ["OPENPOSE_MODELS"] = os.path.join(OPENPOSE_PATH, "models")
OPENPOSE_BIN = os.path.join(OPENPOSE_PATH, "build", "examples", "openpose", "openpose.bin")

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

def extract_keypoints(frame):
    """Runs OpenPose on a frame and extracts keypoints."""
    temp_image_path = "temp_frame.jpg"
    cv2.imwrite(temp_image_path, frame)  # Save frame to file

    # Run OpenPose on the saved image
    command = f"{OPENPOSE_BIN} --image_dir . --write_json . --display 0 --render_pose 0"
    os.system(command)

    # Read OpenPose JSON output
    try:
        with open("temp_frame_keypoints.json", "r") as f:
            import json
            keypoints_data = json.load(f)
            if keypoints_data["people"]:
                return keypoints_data["people"][0]["pose_keypoints_2d"]
    except FileNotFoundError:
        return []

    return []

@app.get("/keypoints")
async def get_keypoints():
    ret, frame = cap.read()
    if not ret:
        return JSONResponse(content={"error": "Unable to access webcam"}, status_code=500)

    keypoints = extract_keypoints(frame)

    # Save to MongoDB
    document = {
        "timestamp": datetime.datetime.utcnow(),
        "keypoints": keypoints
    }
    #collection.insert_one(document)  # Store keypoints in MongoDB

    return {"message": "Keypoints saved", "keypoints": keypoints}

@app.get("/video_feed")
async def video_feed():
    def generate_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run OpenPose to extract and draw keypoints
            keypoints = extract_keypoints(frame)

            # Overlay keypoints on frame
            for i in range(0, len(keypoints), 3):
                x, y, c = keypoints[i], keypoints[i + 1], keypoints[i + 2]
                if c > 0:  # Confidence check
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.on_event("shutdown")
def shutdown_event():
    cap.release()
    client.close()  # Close MongoDB connection

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

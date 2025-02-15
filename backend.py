from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import cv2
import mediapipe as mp
import datetime

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

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)  # Open webcam

@app.get("/keypoints")
async def get_keypoints():
    ret, frame = cap.read()
    if not ret:
        return JSONResponse(content={"error": "Unable to access webcam"}, status_code=500)

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    # Extract keypoints
    keypoints = []
    if result.pose_landmarks:
        for i, landmark in enumerate(result.pose_landmarks.landmark):
            keypoints.append({
                "id": i,
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z
            })

    # Save to MongoDB
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

            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)

            if result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# Gracefully release resources on shutdown
@app.on_event("shutdown")
def shutdown_event():
    cap.release()
    client.close()  # Close MongoDB connection

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

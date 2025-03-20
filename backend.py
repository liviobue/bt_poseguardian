from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import cv2
import mediapipe as mp
import datetime
import numpy as np

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

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)  # Open webcam

def is_hand_open(hand_landmarks):
    """Detects if the hand is fully open by checking if all fingers are straight."""
    
    # Define landmark indices for fingertips and base joints
    FINGERTIPS = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    BASE_JOINTS = [6, 10, 14, 18]  # Joints before fingertips
    
    for tip, base in zip(FINGERTIPS, BASE_JOINTS):
        tip_y = hand_landmarks.landmark[tip].y
        base_y = hand_landmarks.landmark[base].y
        
        # If any fingertip is lower (higher Y) than its base, the hand is not open
        if tip_y > base_y:
            return False

    return True  # All fingers are extended

@app.get("/keypoints")
async def get_keypoints():
    ret, frame = cap.read()
    if not ret:
        return JSONResponse(content={"error": "Unable to access webcam"}, status_code=500)

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    keypoints = []
    recognized = False

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            hand_keypoints = []
            for i, landmark in enumerate(hand_landmarks.landmark):
                hand_keypoints.append({
                    "id": i,
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z
                })

            keypoints.append({"keypoints": hand_keypoints})

            # Check if hand is open
            if is_hand_open(hand_landmarks):
                recognized = True

    # Save to MongoDB
    document = {
        "timestamp": datetime.datetime.utcnow(),
        "hands": keypoints,
        "recognized": recognized
    }
    # collection.insert_one(document)  # Uncomment to store data

    return {"recognized": recognized, "hands": keypoints}

@app.get("/video_feed")
async def video_feed():
    def generate_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)
            recognized = False

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    if is_hand_open(hand_landmarks):
                        recognized = True

            # Overlay text
            text = "Recognized" if recognized else "Not Recognized"
            color = (0, 255, 0) if recognized else (0, 0, 255)
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

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

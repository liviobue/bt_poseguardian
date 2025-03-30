from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import cv2
import mediapipe as mp
import datetime
import numpy as np
from collections import deque
import time
from typing import Dict
import uuid

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

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

# Define the correct sequence: Index → Middle → Ring → Pinky
CORRECT_SEQUENCE = [8, 12, 16, 20, 16, 12, 8]
SEQUENCE_TIMEOUT = 2.0  # seconds to complete the full sequence
TOUCH_THRESHOLD = 0.05  # Distance threshold for touch detection
MIN_FRAMES_FOR_DETECTION = 5

class GestureTracker:
    """Tracks gesture state for a single client/session"""
    def __init__(self):
        self.history_buffer = deque(maxlen=20)
        self.touch_sequence = []
        self.last_touch_time = None
        self.last_recognized_time = 0
        self.recognition_cooldown = 0.5  # 500ms cooldown after recognition
    
    def update_hand_history(self, hand_landmarks):
        """Stores thumb & finger positions in a buffer for tracking gestures."""
        thumb_tip = hand_landmarks.landmark[4]
        fingers = [hand_landmarks.landmark[i] for i in CORRECT_SEQUENCE]

        hand_state = {
            "thumb": (thumb_tip.x, thumb_tip.y),
            "fingers": [(f.x, f.y) for f in fingers],
            "timestamp": time.time()
        }
        self.history_buffer.append(hand_state)
    
    def is_thumb_touching_fingers_sequentially(self):
        """Detects if the thumb touches all four fingers in the correct order with timeout."""
        current_time = time.time()
        
        # If in cooldown period after recognition, skip detection
        if (current_time - self.last_recognized_time) < self.recognition_cooldown:
            return False

        # Reset if sequence times out
        if self.last_touch_time and (current_time - self.last_touch_time) > SEQUENCE_TIMEOUT:
            self._reset_tracking()
            return False

        if len(self.history_buffer) < MIN_FRAMES_FOR_DETECTION:
            return False

        last_state = self.history_buffer[-1]
        thumb_x, thumb_y = last_state["thumb"]
        fingers = last_state["fingers"]

        for i, finger_pos in enumerate(fingers):
            distance = np.linalg.norm(np.array([thumb_x, thumb_y]) - np.array(finger_pos))
            
            if distance < TOUCH_THRESHOLD:
                if CORRECT_SEQUENCE[i] not in self.touch_sequence:
                    if not self.touch_sequence or CORRECT_SEQUENCE[i] == CORRECT_SEQUENCE[len(self.touch_sequence)]:
                        self.touch_sequence.append(CORRECT_SEQUENCE[i])
                        self.last_touch_time = current_time
                    else:
                        self._reset_tracking()

        # Check for complete sequence
        if self.touch_sequence == CORRECT_SEQUENCE:
            self._reset_tracking()
            self.last_recognized_time = current_time
            return True
        
        return False
    
    def _reset_tracking(self):
        """Immediately reset all tracking state"""
        self.touch_sequence = []
        self.last_touch_time = None
    
    def is_gesture_active(self, current_time):
        """Check if gesture was recently recognized (within 4 seconds)"""
        return (current_time - self.last_recognized_time) <= 4.0

# Track sessions by client ID (in a real app, use proper session management)
sessions: Dict[str, GestureTracker] = {}

def is_hand_open(hand_landmarks):
    """Check if hand is fully open (all fingers extended)"""
    # Implement your hand open detection logic here
    # This is a placeholder - you'll need to implement the actual detection
    return False

@app.get("/keypoints")
async def get_keypoints(request: Request):
    client_id = request.headers.get("X-Client-ID", str(uuid.uuid4()))
    
    if client_id not in sessions:
        sessions[client_id] = GestureTracker()
    
    tracker = sessions[client_id]
    
    ret, frame = cap.read()
    if not ret:
        return JSONResponse(content={"error": "Unable to access webcam"}, status_code=500)

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
            tracker.update_hand_history(hand_landmarks)
            
            if is_hand_open(hand_landmarks):
                recognized = True

    document = {
        "timestamp": datetime.datetime.utcnow(),
        "client_id": client_id,
        "hands": keypoints,
        "recognized": recognized
    }
    # collection.insert_one(document)  # Uncomment to store data

    return {"recognized": recognized, "hands": keypoints, "client_id": client_id}

@app.get("/video_feed")
async def video_feed(request: Request):
    client_id = request.headers.get("X-Client-ID", str(uuid.uuid4()))
    
    if client_id not in sessions:
        sessions[client_id] = GestureTracker()
    
    tracker = sessions[client_id]

    def generate_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            current_time = time.time()

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    tracker.update_hand_history(hand_landmarks)
                    
                    if tracker.is_thumb_touching_fingers_sequentially():
                        pass  # Recognition is already handled in the method

            # Determine if message should be displayed
            if tracker.is_gesture_active(current_time):
                text = "Gesture Recognized"
                color = (0, 255, 0)
            else:
                text = "No Gesture"
                color = (0, 0, 255)

            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"X-Client-ID": client_id}
    )

# Clean up old sessions periodically
async def cleanup_sessions():
    while True:
        await asyncio.sleep(60)  # Cleanup every minute
        current_time = time.time()
        to_delete = []
        
        for client_id, tracker in sessions.items():
            # Remove sessions inactive for more than 5 minutes
            if (current_time - tracker.last_recognized_time) > 300:
                to_delete.append(client_id)
        
        for client_id in to_delete:
            del sessions[client_id]

# Start cleanup task on app startup
@app.on_event("startup")
async def startup_event():
    import asyncio
    asyncio.create_task(cleanup_sessions())

@app.on_event("shutdown")
def shutdown_event():
    cap.release()
    client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
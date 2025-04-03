from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import cv2
import mediapipe as mp
import datetime
import numpy as np
import tempfile
import time
import uuid
from google.api_core import exceptions as google_exceptions
import google.generativeai as genai
import os
from dotenv import load_dotenv
from config import Config

# Validate config on startup
try:
    Config.validate()
except ValueError as e:
    print(f"❌ Configuration error: {str(e)}")
    # Exit if running in production
    if not os.getenv("DEV_MODE"):
        raise

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["http://localhost:3000"] for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Connection
client = MongoClient(Config.MONGO_URI)
db = client["bt_poseguardian"]
collection = db["data"]

# Google API Connection
genai.configure(api_key=Config.GOOGLE_API_KEY)
PROMPT = "Analyze this video for rehabilitation movements."
MAX_FILE_SIZE = 20 * 1024 * 1024
SUPPORTED_MIME_TYPES = [
    'video/mp4',
    #'video/webm',
    #'video/quicktime'
]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)  # Open webcam

def is_hand_open(hand_landmarks):
    """Checks if all fingers (including thumb) are fully extended (no bending)."""
    
    # Landmark indices
    FINGERTIPS = [4, 8, 12, 16, 20]   # Thumb, Index, Middle, Ring, Pinky
    MIDDLE_JOINTS = [3, 7, 11, 15, 19] # PIP joints (middle of fingers)
    BASE_JOINTS = [2, 6, 10, 14, 18]   # MCP joints (base of fingers)

    tolerance = 0.01  # Small allowed deviation for natural variations

    for tip, middle, base in zip(FINGERTIPS, MIDDLE_JOINTS, BASE_JOINTS):
        tip_y = hand_landmarks.landmark[tip].y
        middle_y = hand_landmarks.landmark[middle].y
        base_y = hand_landmarks.landmark[base].y

        # For thumb, we need to check x coordinate instead of y (thumb moves differently)
        if tip == 4:  # Thumb tip
            tip_x = hand_landmarks.landmark[tip].x
            middle_x = hand_landmarks.landmark[middle].x
            base_x = hand_landmarks.landmark[base].x
            
            # Thumb is extended if tip is further out than middle joint (x coordinate)
            if not (tip_x > middle_x > base_x):
                return False
            
            # Check if thumb is straight
            expected_middle_x = (tip_x + base_x) / 2
            if abs(middle_x - expected_middle_x) > tolerance:
                return False
        else:
            # For other fingers, check y coordinate as before
            if not (tip_y < middle_y < base_y):  
                return False

            # Ensure fingers are nearly straight
            expected_middle_y = (tip_y + base_y) / 2
            if abs(middle_y - expected_middle_y) > tolerance:
                return False  

    return True  # All fingers including thumb are straight

def extract_hand_keypoints(frame):
    """Extract hand keypoints from a single frame"""
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

            # Check if hand is fully open
            if is_hand_open(hand_landmarks):
                recognized = True

    return {"recognized": recognized, "hands": keypoints}

def process_video_for_keypoints(video_path, sampling_rate=5):
    """
    Process video file and extract hand keypoints at specified sampling rate
    
    Args:
        video_path: Path to the video file
        sampling_rate: Extract keypoints every N frames
        
    Returns:
        Dictionary with video metadata and frames array
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    # Video metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Prepare document structure
    frames_data = []
    frame_count = 0
    hands_detected_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every Nth frame to improve performance
        if frame_count % sampling_rate == 0:
            keypoint_data = extract_hand_keypoints(frame)
            
            # Calculate timestamp based on frame number and fps
            frame_time = frame_count / fps if fps > 0 else 0
            
            # Add frame data
            frame_data = {
                "frame_number": frame_count,
                "video_time": frame_time,
                "hands": keypoint_data["hands"],
                "recognized": keypoint_data["recognized"]
            }
            
            frames_data.append(frame_data)
            
            # Count frames with hands
            if keypoint_data["hands"]:
                hands_detected_count += 1
            
        frame_count += 1
    
    cap.release()
    
    # Return full document structure
    video_data = {
        "video_id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.utcnow(),
        "metadata": {
            "fps": fps,
            "total_frames": total_frames,
            "processed_frames": len(frames_data),
            "sampling_rate": sampling_rate,
            "duration": duration,
            "width": width,
            "height": height,
            "hands_detected_count": hands_detected_count
        },
        "frames": frames_data
    }
    
    return video_data

@app.get("/keypoints")
async def get_keypoints():
    ret, frame = cap.read()
    if not ret:
        return JSONResponse(content={"error": "Unable to access webcam"}, status_code=500)

    keypoint_data = extract_hand_keypoints(frame)
    
    # Save to MongoDB
    document = {
        "timestamp": datetime.datetime.utcnow(),
        "hands": keypoint_data["hands"],
        "recognized": keypoint_data["recognized"]
    }
    # collection.insert_one(document)  # Uncomment to store data

    return keypoint_data

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

@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    # Check file size
    video.file.seek(0, 2)
    file_size = video.file.tell()
    video.file.seek(0)
    if file_size > MAX_FILE_SIZE:
        return {"response": "Error: File too large (max 20MB)"}

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        # Extract keypoints from the video with optimized MongoDB structure
        video_data = process_video_for_keypoints(tmp_path)
        
        # Store in MongoDB if data was extracted
        mongo_id = None
        if video_data and video_data["frames"]:
            result = collection.insert_one(video_data)
            mongo_id = str(result.inserted_id)
            
            keypoints_summary = {
                "saved": True,
                "video_id": video_data["video_id"],
                "mongo_id": mongo_id,
                "total_frames_processed": len(video_data["frames"]),
                "frames_with_hands": video_data["metadata"]["hands_detected_count"],
                "video_duration": video_data["metadata"]["duration"]
            }
        else:
            keypoints_summary = {
                "saved": False,
                "reason": "No hand keypoints detected in video"
            }
        
        # Upload to Gemini for analysis
        file_obj = genai.upload_file(path=tmp_path, mime_type=video.content_type)
        
        # Add delay to ensure file is ready
        time.sleep(5)  # Wait 5 seconds for file to become active

        # Generate response
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            contents=[PROMPT, file_obj],
            generation_config={'max_output_tokens': 2048}
        )
        
        # Return combined response with Gemini analysis and keypoint summary
        return {
            "response": response.text,
            "keypoints_extraction": keypoints_summary
        }
        
    except google_exceptions.FailedPrecondition as e:
        return {"response": f"Error: File not ready for processing. Try again later. Details: {str(e)}"}
    except genai.types.BlockedPromptException:
        return {"response": "Error: Content blocked by safety filters"}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}
    finally:
        # Clean up temp file
        import os
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Gracefully release resources on shutdown
@app.on_event("shutdown")
def shutdown_event():
    cap.release()
    client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
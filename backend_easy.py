from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import datetime
import numpy as np
import tempfile
import time
import uuid
import json
from google.api_core import exceptions as google_exceptions
import google.generativeai as genai
import os
from dotenv import load_dotenv
from config import Config
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import the gesture recognition module
from gestures import GestureRecognizer

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
# Update these variables at the top where they're defined
VIDEO_PROMPT = "Analyze this rehabilitation exercise video. Identify specific movements, potential therapeutic applications, and exercise type. Focus on hand movements if visible. Do not use any Markdown formatting like bold, italics, or bullet points. Reply in plain text only."
KEYPOINTS_PROMPT = "Analyze these hand keypoints data for rehabilitation exercises. Based on the movement patterns, identify the specific exercise type, therapeutic applications, and key movement characteristics. Do not use any Markdown formatting like bold, italics, or bullet points. Reply in plain text only."
MAX_FILE_SIZE = 20 * 1024 * 1024
SUPPORTED_MIME_TYPES = [
    'video/mp4',
    #'video/webm',
    #'video/quicktime'
]

# Initialize Gesture Recognizer
gesture_recognizer = GestureRecognizer()
cap = cv2.VideoCapture(0)  # Open webcam

def calculate_similarity(text1, text2):
    """
    Calculate cosine similarity between two text strings
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
        
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    try:
        # Create TF-IDF features
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except Exception as e:
        print(f"Error calculating similarity: {str(e)}")
        return 0.0

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
            keypoint_data = gesture_recognizer.extract_hand_keypoints(frame)
            
            # Calculate timestamp based on frame number and fps
            frame_time = frame_count / fps if fps > 0 else 0
            
            # Add frame data
            frame_data = {
                "frame_number": frame_count,
                "video_time": frame_time,
                "hands": keypoint_data["hands"],
                "recognized": keypoint_data["recognized"],
                "gestures": keypoint_data["gestures"]
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

def prepare_keypoints_for_gemini(video_data):
    """
    Prepare a text representation of the keypoints data for Gemini analysis
    
    Args:
        video_data: The video data document with frames and keypoints
        
    Returns:
        Formatted text description of the keypoints
    """
    if not video_data or not video_data.get("frames"):
        return "No keypoint data available for analysis."
    
    # Prepare metadata section
    metadata = video_data["metadata"]
    text_data = [
        "# Hand Keypoints Data Analysis",
        f"Video duration: {metadata['duration']:.2f} seconds",
        f"Frames analyzed: {metadata['processed_frames']} out of {metadata['total_frames']}",
        f"Frames with hands detected: {metadata['hands_detected_count']}",
        f"Video dimensions: {metadata['width']}x{metadata['height']} at {metadata['fps']:.2f} FPS",
        "\n## Frame-by-Frame Keypoint Data:"
    ]
    
    # Track detected gestures
    gesture_counts = {}
    
    # Limit to a reasonable number of frames to not exceed token limits
    frames_to_include = min(30, len(video_data["frames"]))
    step = max(1, len(video_data["frames"]) // frames_to_include)
    
    for i in range(0, len(video_data["frames"]), step):
        if i >= len(video_data["frames"]):
            break
            
        frame = video_data["frames"][i]
        frame_text = [
            f"\nFrame {frame['frame_number']} (Time: {frame['video_time']:.2f}s):"
        ]
        
        # Add detected gestures to frame text
        if "gestures" in frame and frame["gestures"]:
            frame_text.append(f"  Detected gestures: {', '.join(frame['gestures'])}")
            
            # Count gesture occurrences for summary
            for gesture in frame["gestures"]:
                if gesture in gesture_counts:
                    gesture_counts[gesture] += 1
                else:
                    gesture_counts[gesture] = 1
        
        if not frame["hands"]:
            frame_text.append("  No hands detected")
        else:
            for hand_idx, hand in enumerate(frame["hands"]):
                frame_text.append(f"  Hand #{hand_idx+1}:")
                
                # Include key landmark positions (not all 21 points to save space)
                keypoints = hand["keypoints"]
                key_points = {
                    0: "Wrist",
                    4: "Thumb tip", 
                    8: "Index finger tip",
                    12: "Middle finger tip",
                    16: "Ring finger tip",
                    20: "Pinky tip"
                }
                
                for point_id, name in key_points.items():
                    for kp in keypoints:
                        if kp["id"] == point_id:
                            frame_text.append(f"    {name}: x={kp['x']:.3f}, y={kp['y']:.3f}, z={kp['z']:.3f}")
                            break
        
        text_data.extend(frame_text)
    
    # Add summary information about hand movement and gestures
    if metadata['hands_detected_count'] > 0:
        text_data.append("\n## Movement Analysis:")
        text_data.append("Hand positions throughout the video can indicate specific exercise patterns.")
        
        # Add gesture summary if any were detected
        if gesture_counts:
            text_data.append("\n## Gesture Summary:")
            for gesture, count in gesture_counts.items():
                percentage = (count / frames_to_include) * 100
                text_data.append(f"  {gesture}: detected in {count} frames ({percentage:.1f}% of analyzed frames)")
            
        text_data.append("\nPlease analyze these keypoints to determine potential rehabilitation exercises.")
    
    return "\n".join(text_data)

@app.get("/keypoints")
async def get_keypoints():
    ret, frame = cap.read()
    if not ret:
        return JSONResponse(content={"error": "Unable to access webcam"}, status_code=500)

    keypoint_data = gesture_recognizer.extract_hand_keypoints(frame)
    
    # Save to MongoDB
    document = {
        "timestamp": datetime.datetime.utcnow(),
        "hands": keypoint_data["hands"],
        "recognized": keypoint_data["recognized"],
        "gestures": keypoint_data["gestures"]
    }
    # collection.insert_one(document)  # Uncomment to store data

    return keypoint_data

@app.get("/video_feed")
async def video_feed():
    def generate_frames():
        for frame, gestures in gesture_recognizer.process_video(cap):
            # Overlay text
            recognized = len(gestures) > 0
            status_text = "Recognized" if recognized else "Not Recognized"
            color = (0, 255, 0) if recognized else (0, 0, 255)
            cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
            # Add detected gesture names
            if gestures:
                gesture_text = ", ".join(gestures)
                cv2.putText(frame, gesture_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
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
        keypoints_text = None
        if video_data and video_data["frames"]:
            result = collection.insert_one(video_data)
            mongo_id = str(result.inserted_id)
            
            # Prepare keypoints text for Gemini
            keypoints_text = prepare_keypoints_for_gemini(video_data)
            
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
        
        # Initialize response variables
        video_analysis = "No video analysis available."
        keypoints_analysis = "No keypoints analysis available."
        
        # Get Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Store prompts for sending to frontend
        full_video_prompt = VIDEO_PROMPT
        full_keypoints_prompt = f"{KEYPOINTS_PROMPT}\n\n{keypoints_text}" if keypoints_text else KEYPOINTS_PROMPT
        
        # 1. Analyze video with Gemini
        try:
            file_obj = genai.upload_file(path=tmp_path, mime_type=video.content_type)
            # Add delay to ensure file is ready
            time.sleep(5)  # Wait 5 seconds for file to become active
            
            video_response = model.generate_content(
                contents=[VIDEO_PROMPT, file_obj],
                generation_config={'max_output_tokens': 2048}
            )
            video_analysis = video_response.text
        except Exception as e:
            video_analysis = f"Error analyzing video: {str(e)}"
        
        # 2. Analyze keypoints with Gemini
        try:
            if keypoints_text:
                keypoints_response = model.generate_content(
                    contents=[KEYPOINTS_PROMPT, keypoints_text],
                    generation_config={'max_output_tokens': 2048}
                )
                keypoints_analysis = keypoints_response.text
        except Exception as e:
            keypoints_analysis = f"Error analyzing keypoints: {str(e)}"
        
         # Calculate similarity score
        similarity_score = calculate_similarity(video_analysis, keypoints_analysis)

        # Return combined response with both analyses, similarity score, and prompts
        return {
            "video_analysis": video_analysis,
            "keypoints_analysis": keypoints_analysis,
            "similarity_score": round(similarity_score * 100, 2),
            "prompts": {
                "video_prompt": full_video_prompt,
                "keypoints_prompt": full_keypoints_prompt
            }
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
    gesture_recognizer.close()
    client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
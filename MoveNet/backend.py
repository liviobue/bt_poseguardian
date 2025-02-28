from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import datetime

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["http://localhost:3000"] for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load MoveNet Model
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

# Open webcam
cap = cv2.VideoCapture(0)

def detect_keypoints(frame):
    """ Runs MoveNet on a given frame and returns keypoints """
    img = cv2.resize(frame, (192, 192))  # Resize to 192x192
    img = np.asarray(img, dtype=np.int32)  # Ensure int32 dtype
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Run MoveNet model
    outputs = movenet.signatures["serving_default"](tf.constant(img))
    keypoints = outputs["output_0"].numpy().reshape(-1, 3)  # Shape: (17,3)

    # Convert keypoints to a readable format
    h, w, _ = frame.shape
    keypoints_list = []
    for i, (y, x, confidence) in enumerate(keypoints):
        keypoints_list.append({
            "id": i,
            "x": float(x) * w,
            "y": float(y) * h,
            "confidence": float(confidence)
        })

    return keypoints_list

@app.get("/keypoints")
async def get_keypoints():
    """ Fetches keypoints from MoveNet model """
    ret, frame = cap.read()
    if not ret:
        return JSONResponse(content={"error": "Unable to access webcam"}, status_code=500)

    keypoints = detect_keypoints(frame)

    # Save keypoints to MongoDB (if needed)
    document = {
        "timestamp": datetime.datetime.utcnow(),
        "keypoints": keypoints
    }
    # collection.insert_one(document)  # Uncomment if using MongoDB

    return {"message": "Keypoints extracted", "keypoints": keypoints}

@app.get("/video_feed")
async def video_feed():
    """ Streams webcam with keypoints overlay """
    def generate_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            keypoints = detect_keypoints(frame)

            # Draw keypoints on the frame
            for kp in keypoints:
                x, y = int(kp["x"]), int(kp["y"])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green dots

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# Gracefully release webcam
@app.on_event("shutdown")
def shutdown_event():
    cap.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

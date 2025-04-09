# 🧠 bt_poseguardian

**bt_poseguardian** is a full-stack real-time gesture recognition system for tracking thumb-to-finger touch gestures using webcam input. It's ideal for physical rehab, gesture interfaces, and movement tracking.

## 💡 Features
- FastAPI backend with MediaPipe for hand gesture recognition
- React frontend with live video view and gesture status
- Real-time webcam gesture recognition
- MongoDB integration for data logging
- Dockerized for easy deployment

## 🛠 Tech Stack
- **Backend**: Python, FastAPI, OpenCV, MediaPipe
- **Frontend**: React
- **Database**: MongoDB
- **Deployment**: Docker, Docker Compose

## 🚀 How to Run Locally

### Backend (FastAPI)
```bash
uvicorn main:app --reload

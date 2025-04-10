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

## 🚀 Getting Started with Docker

### 📦 Prerequisites

- [Docker](https://www.docker.com/) installed
- [Docker Compose](https://docs.docker.com/compose/) installed

### 📁 Example `.env` file

Create a `.env` file in the root of your project and add (not committed to Git):

```env
GOOGLE_API_KEY=your_google_api_key_here
MONGO_URI=your_mongo_uri_here
```

### 🛠 Build Docker Images Locally

To build the Docker images for both the backend and frontend locally, run:

```bash
docker-compose build
```

### ▶️ Run the Application  

Run this command to start the application

```bash  
docker-compose up  
```

This will start:
- Backend: http://localhost:8000
- Frontend: http://localhost:3000

### 🛑 Stop the App
Press Ctrl+C in your terminal, then:

```env
docker-compose down
```
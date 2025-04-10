import React, { useRef, useEffect, useState } from 'react';
import axios from 'axios';

const WebcamCapture = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [handData, setHandData] = useState(null);
  const [gestures, setGestures] = useState([]);
  const [isRecognized, setIsRecognized] = useState(false);
  const [error, setError] = useState(null);
  const [showKeypoints, setShowKeypoints] = useState(true);

  // Initialize webcam when component mounts
  useEffect(() => {
    let videoStream = null;

    const setupWebcam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: "user"
          } 
        });
        
        videoStream = stream;
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setIsCapturing(true);
        }
      } catch (err) {
        setError(`Error accessing webcam: ${err.message}`);
        console.error("Error accessing webcam:", err);
      }
    };

    setupWebcam();
    
    // Clean up function to stop all tracks when component unmounts
    return () => {
      if (videoStream) {
        const tracks = videoStream.getTracks();
        tracks.forEach(track => track.stop());
      }
    };
  }, []);

  // Set up canvas and frame capture
  useEffect(() => {
    if (!isCapturing) return;

    const captureInterval = setInterval(() => {
      sendFrameToBackend();
    }, 100); // Send frame every 100ms (10fps)

    return () => clearInterval(captureInterval);
  }, [isCapturing]);

  // Draw keypoints on overlay canvas whenever hand data updates
  useEffect(() => {
    if (!handData || !overlayCanvasRef.current || !videoRef.current || !showKeypoints) {
      // If keypoints shouldn't be shown, clear the canvas
      if (overlayCanvasRef.current) {
        const ctx = overlayCanvasRef.current.getContext('2d');
        ctx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
      }
      return;
    }
    
    const canvas = overlayCanvasRef.current;
    const video = videoRef.current;
    
    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw keypoints and connections
    drawKeypoints(ctx, handData);
    
  }, [handData, showKeypoints]);

  // Function to draw keypoints on canvas
  const drawKeypoints = (ctx, hands) => {
    if (!hands || hands.length === 0) return;
    
    hands.forEach(hand => {
      const keypoints = hand.keypoints;
      
      // Draw points
      keypoints.forEach(point => {
        // Convert normalized coordinates to pixel coordinates
        const x = point.x * ctx.canvas.width;
        const y = point.y * ctx.canvas.height;
        
        // Draw circle for each keypoint
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = '#00FF00';
        ctx.fill();
      });
      
      // Draw connections between points (simplified hand skeleton)
      const connections = [
        // Thumb
        [0, 1], [1, 2], [2, 3], [3, 4],
        // Index finger
        [0, 5], [5, 6], [6, 7], [7, 8],
        // Middle finger
        [0, 9], [9, 10], [10, 11], [11, 12],
        // Ring finger
        [0, 13], [13, 14], [14, 15], [15, 16],
        // Pinky
        [0, 17], [17, 18], [18, 19], [19, 20],
        // Palm
        [0, 5], [5, 9], [9, 13], [13, 17]
      ];
      
      connections.forEach(([i, j]) => {
        const startPoint = keypoints.find(p => p.id === i);
        const endPoint = keypoints.find(p => p.id === j);
        
        if (startPoint && endPoint) {
          ctx.beginPath();
          ctx.moveTo(startPoint.x * ctx.canvas.width, startPoint.y * ctx.canvas.height);
          ctx.lineTo(endPoint.x * ctx.canvas.width, endPoint.y * ctx.canvas.height);
          ctx.strokeStyle = '#FFFF00';
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      });
    });
  };

  // Function to capture and send video frame to backend
  const sendFrameToBackend = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw the current video frame on the canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to blob
    try {
      canvas.toBlob(async (blob) => {
        if (!blob) return;

        // Create form data with the image
        const formData = new FormData();
        formData.append('frame', blob, 'frame.jpg');

        // Send to backend
        try {
          const response = await axios.post(
            `${window.location.protocol}//${window.location.hostname}:8000/process_frame`, 
            formData,
            { headers: { 'Content-Type': 'multipart/form-data' } }
          );

          // Update state with keypoint data
          setHandData(response.data.hands);
          setGestures(response.data.gestures || []);
          setIsRecognized(response.data.recognized || false);
        } catch (err) {
          console.error("Error sending frame to backend:", err);
        }
      }, 'image/jpeg', 0.8);
    } catch (err) {
      console.error("Error creating blob:", err);
    }
  };

  // Toggle keypoints visibility
  const toggleKeypoints = () => {
    setShowKeypoints(!showKeypoints);
  };

  return (
    <div style={containerStyle}>
      <h2>Pose Estimation</h2>
      
      {/* Display any errors */}
      {error && <div style={{ color: 'red', marginBottom: '10px' }}>{error}</div>}
      
      {/* Toggle button for keypoints */}
      <button 
        onClick={toggleKeypoints} 
        style={buttonStyle}
      >
        {showKeypoints ? 'Hide Keypoints' : 'Show Keypoints'}
      </button>
      
      {/* Webcam Video Display */}
      <div style={videoContainerStyle}>
        {/* Video from webcam */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={videoStyle}
          onCanPlay={() => videoRef.current?.play()}
        />
        
        {/* Overlay canvas for keypoints */}
        <canvas 
          ref={overlayCanvasRef}
          style={{
            ...overlayCanvasStyle,
            display: showKeypoints && handData && handData.length > 0 ? 'block' : 'none'
          }}
        />
        
        {/* Recognition status overlay */}
        <div style={{
          ...statusOverlayStyle,
          backgroundColor: isRecognized ? 'rgba(0, 128, 0, 0.5)' : 'rgba(255, 0, 0, 0.5)'
        }}>
          <div style={{ fontWeight: 'bold' }}>
            {isRecognized ? 'Recognized' : 'Not Recognized'}
          </div>
          {gestures.length > 0 && (
            <div>{gestures.join(', ')}</div>
          )}
        </div>

        {/* Hidden canvas for capturing frames */}
        <canvas ref={canvasRef} style={{ display: 'none' }} />
      </div>
    </div>
  );
};

const containerStyle = {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  padding: '20px',
};

const videoContainerStyle = {
  width: "80%",
  display: "flex",
  justifyContent: "center",
  marginBottom: "20px",
  position: "relative"
};

const videoStyle = {
  width: "100%",
  maxHeight: "400px",
  borderRadius: "10px",
  border: "2px solid #333"
};

const overlayCanvasStyle = {
  position: 'absolute',
  top: 0,
  left: 0,
  width: '100%',
  height: '100%',
  pointerEvents: 'none'
};

const statusOverlayStyle = {
  position: 'absolute',
  top: '20px',
  left: '20px',
  color: 'white',
  padding: '10px',
  borderRadius: '5px',
  fontFamily: 'Arial, sans-serif',
  zIndex: 10
};

const buttonStyle = {
  padding: '8px 15px',
  marginBottom: '15px',
  backgroundColor: '#4285f4',
  color: 'white',
  border: 'none',
  borderRadius: '4px',
  cursor: 'pointer',
  fontWeight: 'bold',
  transition: 'background-color 0.3s'
};

export default WebcamCapture;
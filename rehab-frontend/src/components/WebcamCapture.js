import React, { useEffect, useState } from 'react';
import axios from 'axios';

const WebcamCapture = () => {
  const [keypoints, setKeypoints] = useState([]);
  const videoFeedUrl = "http://127.0.0.1:8000/video_feed";

  useEffect(() => {
    const fetchKeypoints = async () => {
      try {
        const response = await axios.get("http://127.0.0.1:8000/keypoints");
        setKeypoints(response.data.keypoints || []);
      } catch (error) {
        console.error("Error fetching keypoints:", error);
      }
    };

    const intervalId = setInterval(fetchKeypoints, 2000); // Fetch every 500ms
    return () => clearInterval(intervalId);
  }, []);

  return (
    <div style={containerStyle}>
      <h2>Pose Estimation</h2>
      
      {/* Webcam Video Feed */}
      <div style={videoContainerStyle}>
        <img src={videoFeedUrl} alt="Video Feed" style={videoStyle} />
      </div>

      {/* Keypoints Data */}
      <div>
        <h3>Keypoints:</h3>
        <pre style={keypointsStyle}>{JSON.stringify(keypoints, null, 2)}</pre>
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
};

const videoStyle = {
  width: "100%",
  maxHeight: "400px",
  borderRadius: "10px",
  border: "2px solid #333"
};

const keypointsStyle = {
  textAlign: 'left',
  background: '#f4f4f4',
  padding: '10px',
  borderRadius: '5px',
  maxHeight: '300px',
  overflow: 'auto',
  width: '80%',
};

export default WebcamCapture;

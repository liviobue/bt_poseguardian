import React from 'react';

const WebcamCapture = () => {
  const videoFeedUrl = "http://127.0.0.1:8000/video_feed";

  return (
    <div style={containerStyle}>
      <h2>Pose Estimation</h2>
      
      {/* Webcam Video Feed */}
      <div style={videoContainerStyle}>
        <img src={videoFeedUrl} alt="Video Feed" style={videoStyle} />
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

export default WebcamCapture;

import React, { useRef, useState } from 'react';
import Webcam from 'react-webcam';

const WebcamCapture = () => {
  const webcamRef = useRef(null);
  const [capturedImage, setCapturedImage] = useState(null);

  const capture = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    setCapturedImage(imageSrc); // Set the captured image
  };

  const videoConstraints = {
    width: 500,
    height: 200,
    facingMode: "user"
  };

  return (
    <div style={webcamContainerStyle}>
      <h2>Webcam Feed</h2>
      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        videoConstraints={videoConstraints}
      />
      <button onClick={capture} style={buttonStyle}>Capture</button>
      {capturedImage && (
        <div style={{ marginTop: '20px' }}>
          <h3>Captured Image:</h3>
          <img src={capturedImage} alt="Captured" />
        </div>
      )}
    </div>
  );
};

const webcamContainerStyle = {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  border: '2px solid #ddd',
  padding: '20px',
  margin: '10px',
};

const buttonStyle = {
  marginTop: '10px',
  padding: '10px',
  backgroundColor: '#282c34',
  color: 'white',
  border: 'none',
  cursor: 'pointer'
};

export default WebcamCapture;

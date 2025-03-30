import React, { useState, useRef } from 'react';
import ResponseDisplay from './ResponseDisplay';

const VideoInput = () => {
  const [videoFile, setVideoFile] = useState(null);
  const [response, setResponse] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunks = useRef([]);

  const handleFileChange = (event) => {
    setVideoFile(event.target.files[0]);
  };

  const handleUpload = async (file) => {
    const formData = new FormData();
    formData.append('video', file);
  
    try {
      const res = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });
  
      const data = await res.json();
      setResponse(data.response);
    } catch (error) {
      console.error('Error uploading video:', error);
      setResponse('Error processing video.');
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      setIsRecording(true);

      mediaRecorderRef.current = new MediaRecorder(stream);
      mediaRecorderRef.current.ondataavailable = (event) => {
        chunks.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunks.current, { type: 'video/webm' });
        const file = new File([blob], 'webcam_video.mp4', { type: 'video/mp4' });
        chunks.current = [];
        setIsRecording(false);
        handleUpload(file);
      };

      mediaRecorderRef.current.start();

      // Automatically stop recording after 20 seconds
      setTimeout(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
          stopRecording();
        }
      }, 20000);
    } catch (error) {
      console.error('Error accessing webcam:', error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      const stream = videoRef.current.srcObject;
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        videoRef.current.srcObject = null;
      }
    }
  };

  return (
    <div style={containerStyle}>
      <div style={videoStyle}>
        <h2>Video Input</h2>
        <p>Upload a video file or record from the webcam.</p>
        <input type="file" accept="video/*" onChange={handleFileChange} />
        <button onClick={() => handleUpload(videoFile)}>Upload Video</button>
        <br />
        <video ref={videoRef} autoPlay style={videoPreviewStyle} />
        <br />
        <button onClick={startRecording} disabled={isRecording}>Start Recording</button>
        <button onClick={stopRecording} disabled={!isRecording}>Stop Recording</button>
      </div>
      <ResponseDisplay response={response} />
    </div>
  );
};

const containerStyle = {
  position: 'relative',
};

const videoStyle = {
  border: '2px solid #ddd',
  padding: '20px',
  margin: '10px',
  textAlign: 'center',
};

const videoPreviewStyle = {
  width: '300px',
  height: '200px',
  marginTop: '10px',
};

export default VideoInput;

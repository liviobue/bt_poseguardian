import React, { useState, useRef } from 'react';
import ResponseDisplay from './ResponseDisplay';

const VideoInput = () => {
  const [videoFile, setVideoFile] = useState(null);
  const [response, setResponse] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunks = useRef([]);

  // Create a static API URL with port 8000
  const API_URL = `${window.location.protocol}//${window.location.hostname}:8000`;

  const handleFileChange = (event) => {
    setVideoFile(event.target.files[0]);
  };

  const handleUpload = async (file) => {
    if (!file) return;
    
    // Reset state
    setIsLoading(true);
    setUploadProgress(0);
    setResponse(null);
    
    const formData = new FormData();
    formData.append('video', file);

    try {
      // Use the static API_URL with port 8000
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${API_URL}/upload`, true);
      
      // Track upload progress
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const progress = Math.round((event.loaded / event.total) * 100);
          setUploadProgress(progress);
        }
      };
      
      xhr.onload = async () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          // Set loading to 100% when upload is complete but processing
          setUploadProgress(100);
          
          try {
            const data = JSON.parse(xhr.responseText);
            // Only stop loading once we have processed the response
            setResponse(data);
            setIsLoading(false);
          } catch (error) {
            console.error('Error parsing response:', error);
            setResponse('Error processing response.');
            setIsLoading(false);
          }
        } else {
          console.error('Server returned an error:', xhr.status, xhr.responseText);
          setResponse(`Server error: ${xhr.status}`);
          setIsLoading(false);
        }
      };
      
      xhr.onerror = () => {
        console.error('Error uploading video');
        setResponse('Error processing video.');
        setIsLoading(false);
      };
      
      xhr.send(formData);
    } catch (error) {
      console.error('Error uploading video:', error);
      setResponse('Error processing video.');
      setIsLoading(false);
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

  // Message to display during different stages of loading
  const getLoadingMessage = () => {
    if (uploadProgress < 100) {
      return `Uploading: ${uploadProgress}%`;
    } else {
      return "Processing video...";
    }
  };

  return (
    <div style={containerStyle}>
      <div style={videoStyle}>
        <h2>Video Input</h2>
        <p>Upload a video file or record from the webcam.</p>
        <input 
          type="file" 
          accept="video/*" 
          onChange={handleFileChange} 
          disabled={isLoading}
        />
        <button 
          onClick={() => handleUpload(videoFile)} 
          disabled={isLoading || !videoFile}
        >
          {isLoading ? 'Processing...' : 'Upload Video'}
        </button>
        
        {isLoading && (
          <div style={loadingContainerStyle}>
            <div style={loadingBarOuterStyle}>
              <div 
                style={{
                  ...loadingBarInnerStyle,
                  width: `${uploadProgress}%`
                }}
              ></div>
            </div>
            <div style={progressTextStyle}>{getLoadingMessage()}</div>
          </div>
        )}
        
        <br />
        <video ref={videoRef} autoPlay style={videoPreviewStyle} />
        <br />
        <button onClick={startRecording} disabled={isRecording || isLoading}>Start Recording</button>
        <button onClick={stopRecording} disabled={!isRecording || isLoading}>Stop Recording</button>
      </div>
      
      {!isLoading && <ResponseDisplay response={response} />}
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

const loadingContainerStyle = {
  margin: '15px 0',
  textAlign: 'center',
  width: '100%',
};

const loadingBarOuterStyle = {
  width: '100%',
  backgroundColor: '#f0f0f0',
  borderRadius: '4px',
  margin: '5px 0',
  height: '20px',
  overflow: 'hidden',
};

const loadingBarInnerStyle = {
  height: '100%',
  backgroundColor: '#4CAF50',
  borderRadius: '4px',
  transition: 'width 0.3s ease-in-out',
};

const progressTextStyle = {
  fontSize: '14px',
  color: '#555',
};

export default VideoInput;
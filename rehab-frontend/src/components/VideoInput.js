import React from 'react';

const VideoInput = () => {
  return (
    <div style={videoStyle}>
      <h2>Video Input</h2>
      <p>Here you can capture video data or upload a file.</p>
      <input type="file" accept="video/*" />
    </div>
  );
};

const videoStyle = {
  border: '2px solid #ddd',
  padding: '20px',
  margin: '10px',
  textAlign: 'center',
};

export default VideoInput;

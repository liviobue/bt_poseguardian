import React from 'react';

const ResponseDisplay = ({ response }) => {
  // Check if response exists and is an object
  if (!response || typeof response !== 'object') {
    return <div style={containerStyle}>No response data available yet.</div>;
  }

  return (
    <div style={containerStyle}>
      <div style={videoStyle}>
        <h3>Response raw video:</h3>
        <p>{response.video_analysis || 'No video analysis available'}</p>
        <h3>Response Keypoints:</h3>
        <p>{response.keypoints_analysis || 'No keypoints analysis available'}</p>
      </div>
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

export default ResponseDisplay;
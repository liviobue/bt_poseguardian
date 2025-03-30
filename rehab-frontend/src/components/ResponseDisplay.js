import React from 'react';

const ResponseDisplay = ({ response }) => {
  return (
    <div style={containerStyle}>
      <div style={videoStyle}>
        <h3>Response:</h3>
        <p>{response}</p>
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

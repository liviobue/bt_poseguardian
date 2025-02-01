import React from 'react';

const SensorData = () => {
  return (
    <div style={sensorStyle}>
      <h2>Sensor Data</h2>
      <p>Example: Accelerometer readings, Gyroscope data, etc.</p>
    </div>
  );
};

const sensorStyle = {
  border: '2px solid #ddd',
  padding: '20px',
  margin: '10px',
  textAlign: 'center',
};

export default SensorData;

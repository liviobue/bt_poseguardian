import React from 'react';

const ProgressChart = () => {
  return (
    <div style={chartStyle}>
      <h2>Progress Chart</h2>
      <p>A graph showing patient progress over time will be displayed here.</p>
    </div>
  );
};

const chartStyle = {
  border: '2px solid #ddd',
  padding: '20px',
  margin: '10px',
  textAlign: 'center',
};

export default ProgressChart;

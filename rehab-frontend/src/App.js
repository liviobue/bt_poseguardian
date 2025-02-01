import React from 'react';
import Header from './components/Header';
import VideoInput from './components/VideoInput';
import SensorData from './components/SensorData';
import ProgressChart from './components/ProgressChart';
import WebcamCapture from './components/WebcamCapture';
import './App.css';

function App() {
  return (
    <div className="App">
      <Header />
      <div style={mainContainer}>
        <VideoInput />
        <WebcamCapture />
        <SensorData />
        <ProgressChart />
      </div>
    </div>
  );
}

const mainContainer = {
  display: 'grid',
  gridTemplateColumns: '1fr 1fr',
  gap: '20px',
  padding: '20px',
};

export default App;

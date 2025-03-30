import React from 'react';
import Header from './components/Header';
import VideoInput from './components/VideoInput';
import WebcamCapture from './components/WebcamCapture';
import './App.css';

function App() {
  return (
    <div className="App">
      <Header />
      <div style={mainContainer}>
        <VideoInput />
        <WebcamCapture />
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

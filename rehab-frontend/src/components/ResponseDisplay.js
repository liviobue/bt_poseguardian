import React, { useEffect } from 'react';

const ResponseDisplay = ({ response }) => {
  // Use Effect must be placed before any conditional returns
  useEffect(() => {
    if (response && response.prompts) {
      console.log('VIDEO PROMPT:');
      console.log(response.prompts.video_prompt);
      
      console.log('KEYPOINTS PROMPT (including full keypoints data):');
      console.log(response.prompts.keypoints_prompt);
    }
  }, [response]);

  // Check if response exists and is an object
  if (!response || typeof response !== 'object') {
    return <div style={containerStyle}>No response data available yet.</div>;
  }

  // Determine color based on similarity score
  const getSimilarityColor = (score) => {
    if (score >= 80) return '#4CAF50'; // Green for high similarity
    if (score >= 50) return '#FFC107'; // Yellow for medium similarity
    return '#F44336'; // Red for low similarity
  }

  const similarityColor = getSimilarityColor(response.similarity_score || 0);

  return (
    <div style={containerStyle}>
      <div style={videoStyle}>
        <h3>Video Analysis:</h3>
        <pre style={{ whiteSpace: 'pre-wrap', wordWrap: 'break-word' }}>
          {response.video_analysis || 'No video analysis available'}
        </pre>
        
        <h3>Keypoints Analysis:</h3>
        <pre style={{ whiteSpace: 'pre-wrap', wordWrap: 'break-word' }}>
          {response.keypoints_analysis || 'No keypoints analysis available'}
        </pre>
        
        {response.similarity_score !== undefined && (
          <div style={similarityContainerStyle}>
            <h3>Similarity Score:</h3>
            <div style={{
              ...similarityScoreStyle,
              backgroundColor: similarityColor
            }}>
              {response.similarity_score}%
            </div>
            <p style={similarityTextStyle}>
              {response.similarity_score >= 80 ? 'High similarity between analyses' : 
              response.similarity_score >= 50 ? 'Moderate similarity between analyses' : 
              'Low similarity between analyses'}
            </p>
          </div>
        )}
        
        {response.prompts && (
          <div style={promptsButtonStyle} onClick={() => {
            console.log('VIDEO PROMPT:');
            console.log(response.prompts.video_prompt);
            
            console.log('KEYPOINTS PROMPT (including full keypoints data):');
            console.log(response.prompts.keypoints_prompt);
          }}>
            Log Prompts to Console
          </div>
        )}
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
  textAlign: 'left',
};

const similarityContainerStyle = {
  marginTop: '20px',
  padding: '10px',
  border: '1px solid #eee',
  borderRadius: '5px',
  backgroundColor: '#f9f9f9',
};

const similarityScoreStyle = {
  display: 'inline-block',
  padding: '8px 16px',
  borderRadius: '20px',
  color: 'white',
  fontWeight: 'bold',
  fontSize: '18px',
  margin: '10px 0',
};

const similarityTextStyle = {
  fontStyle: 'italic',
  color: '#555',
};

const promptsButtonStyle = {
  marginTop: '20px',
  padding: '10px 15px',
  backgroundColor: '#2196F3',
  color: 'white',
  borderRadius: '4px',
  cursor: 'pointer',
  display: 'inline-block',
  fontWeight: 'bold',
}

export default ResponseDisplay;
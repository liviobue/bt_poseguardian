import React from 'react';

const Header = () => {
  return (
    <header style={headerStyle}>
      <h1>Rehabilitation Tracker</h1>
    </header>
  );
};

const headerStyle = {
  backgroundColor: '#282c34',
  padding: '10px',
  color: 'white',
  textAlign: 'center',
};

export default Header;

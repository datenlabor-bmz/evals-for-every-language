import React from 'react';

const Medal = ({ rank }) => {
  const baseMedalStyle = {
    margin: '0px',
    fontWeight: '900',
    fontStretch: '150%',
    fontFamily: 'Inter, -apple-system, sans-serif',
    width: '24px',
    height: '24px',
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '0.95rem',
    lineHeight: '1',
    padding: '0px',
    position: 'relative'
  }
  const medalStyle1 = {
    ...baseMedalStyle,
    color: 'rgb(181, 138, 27)',
    background:
      'linear-gradient(135deg, rgb(255, 247, 224) 0%, rgb(255, 215, 0) 100%)',
    border: '1px solid rgba(212, 160, 23, 0.35)',
    boxShadow: 'rgba(212, 160, 23, 0.8) 1px 1px 0px'
  }
  const medalStyle2 = {
    color: 'rgb(102, 115, 128)',
    background:
      'linear-gradient(135deg, rgb(255, 255, 255) 0%, rgb(216, 227, 237) 100%)',
    border: '1px solid rgba(124, 139, 153, 0.35)',
    boxShadow: 'rgba(124, 139, 153, 0.8) 1px 1px 0px'
  }
  const medalStyle3 = {
    color: 'rgb(184, 92, 47)',
    background:
      'linear-gradient(135deg, rgb(253, 240, 233) 0%, rgb(255, 188, 140) 100%)',
    border: '1px solid rgba(204, 108, 61, 0.35)',
    boxShadow: 'rgba(204, 108, 61, 0.8) 1px 1px 0px'
  }
  const medalStyle = {
    ...baseMedalStyle,
    ...(rank < 4 ? [medalStyle1, medalStyle2, medalStyle3][rank - 1] : {})
  }
  return (
    <div
      style={{
        alignItems: 'center',
        justifyContent: 'center',
        display: 'flex'
      }}
    >
      <div style={medalStyle}>{rank}</div>
    </div>
  )
}

export default Medal;

import React from 'react';
import { Button, CircularProgress } from '@material-ui/core';
import { green } from '@material-ui/core/colors';

const wrapperStyle: React.CSSProperties = {
  position: 'relative',
};

const buttonProgressStyle: React.CSSProperties = {
  color: green[500],
  position: 'absolute',
  top: '50%',
  left: '50%',
  marginTop: -12,
  marginLeft: -12,
};

interface UploadButtonProps {
  onClick: (event: any) => void;
  isActive: boolean;
}

const UploadButton: React.FC<UploadButtonProps> = ({ onClick, isActive }) => {
  return (
    <div>
      <input
        accept="image/*"
        style={{ display: 'none' }}
        id="icon-button-photo"
        onChange={onClick}
        type="file"
        disabled={!isActive}
      />
      <label htmlFor="icon-button-photo">
        <div style={wrapperStyle}>
          <Button variant="contained" component="span" disabled={!isActive}>
            Upload Image
          </Button>
          {!isActive && (
            <CircularProgress size={24} style={buttonProgressStyle} />
          )}
        </div>
      </label>
    </div>
  );
};

export default UploadButton;

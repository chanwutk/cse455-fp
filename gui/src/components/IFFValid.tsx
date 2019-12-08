import React from 'react';

interface IFFValidProps {
  validate: any;
}

const IFFValid: React.FC<IFFValidProps> = ({ validate, children }) => {
  return (
    <>
      {validate !== null ? (
        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
          }}
        >
          {children}
        </div>
      ) : null}
    </>
  );
};

export default IFFValid;

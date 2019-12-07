import React from 'react';
import './App.css';
import UploadButton from './components/UploadButton';
import { CANVAS_MAX_WIDTH, makeRequest, drawArrow } from './utils';
import ExampleImages from './components/ExampleImages';

const rootStyle: React.CSSProperties = {
  width: '100%',
  position: 'absolute',
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
};

const bodyStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  width: CANVAS_MAX_WIDTH + 'px',
  justifyContent: 'center',
};

const blockStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'center',
  paddingTop: '30px',
  paddingBottom: '30px',
};

interface AppState {
  imageData?: string;
  isNormal: null | boolean;
  isUploadButtonActive: boolean;
}

class App extends React.Component<{}, AppState> {
  state: AppState = {
    isNormal: null,
    isUploadButtonActive: true,
  };

  pictureRef: React.RefObject<HTMLCanvasElement> = React.createRef();
  lastArrowRef: React.RefObject<HTMLCanvasElement> = React.createRef();

  handleFileUpload = (event: any) => {
    const file = event.target.files[0];
    const src = URL.createObjectURL(file);
    const image = new Image();
    const canvas = this.pictureRef.current!;
    image.src = src;
    image.onload = () => {
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(image, 0, 0, 224, 224);
      const imageData = canvas.toDataURL('image/jpeg');
      if (imageData !== this.state.imageData) {
        (async () => {
          const init = {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ data: imageData }),
          };
          const output = await makeRequest('classify', c => c, init);

          await this.setState({
            isNormal: output === 'normal',
            isUploadButtonActive: true,
            imageData,
          });

          drawArrow(this.lastArrowRef.current!);
        })();
      }
    };
  };

  render() {
    return (
      <div style={rootStyle}>
        <div style={bodyStyle}>
          <div style={{ marginBottom: 40 }}>
            <h1 style={{ fontSize: '40px', marginBottom: 0 }}>
              TODO: Title Here!
            </h1>
          </div>
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <canvas
              width="224px"
              height="224px"
              ref={this.pictureRef}
              style={{ borderStyle: 'dotted', padding: '2px' }}
            />
          </div>
          <div style={blockStyle}>
            <UploadButton
              onClick={this.handleFileUpload}
              isActive={this.state.isUploadButtonActive}
            />
          </div>
          {this.state.isNormal !== null ? (
            <div style={blockStyle}>
              <canvas ref={this.lastArrowRef} width="80" height="80" />
            </div>
          ) : null}
          <div
            style={{
              display: 'flex',
              justifyContent: 'center',
              fontSize: 50,
              marginBottom: 20,
            }}
          >
            {this.state.isNormal !== null
              ? this.state.isNormal
                ? 'This lung is normal!'
                : 'This lung is infected with pneumonia :('
              : ''}
          </div>
          <div
            style={{
              display: 'flex',
              justifyContent: 'center',
              marginTop: 20,
              marginBottom: 20,
            }}
          >
            {this.state.isNormal !== null
              ? `Example pictures of ${
                  this.state.isNormal
                    ? 'normal lungs'
                    : 'lungs with pneumonia infection'
                }`
              : null}{' '}
          </div>
          {this.state.isNormal !== null ? (
            <ExampleImages isNormal={this.state.isNormal} N={3} />
          ) : null}
        </div>
      </div>
    );
  }
}

export default App;

import React from 'react';
import { makeRequest, writeBase64ToCanvas } from '../utils';

interface ExampleImagesProps {
  isNormal: boolean;
  N: number;
}

class ExampleImages extends React.Component<ExampleImagesProps> {
  canvasRefs: React.RefObject<HTMLCanvasElement>[];

  constructor(props: ExampleImagesProps) {
    super(props);
    this.canvasRefs = [];
    for (let i = 0; i < this.props.N; i++) {
      this.canvasRefs.push(React.createRef());
    }
  }

  componentDidMount = () => {
    setTimeout(async () => {
      const exampleImages: string[] = await makeRequest(
        `example-images/${this.props.isNormal}/${this.props.N}`,
        JSON.parse
      );
      for (let i = 0; i < this.props.N; i++) {
        const canvas = this.canvasRefs[i].current!;
        writeBase64ToCanvas(canvas, exampleImages[i], 500);
      }
    }, 10);
  };

  render() {
    return (
      <div style={{ display: 'flex', flexDirection: 'row' }}>
        {this.canvasRefs.map((canvasRef, idx) => (
          <canvas
            key={`key-${idx}`}
            ref={canvasRef}
            width="500"
            height="500"
          ></canvas>
        ))}
      </div>
    );
  }
}

export default ExampleImages;

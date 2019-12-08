import React from 'react';
import { makeRequest, writeBase64ToCanvas } from '../utils';

const SIZE = 300;

interface ExampleImagesProps {
  isNormal: boolean;
  N: number;
}

const canvasLayoutStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'row',
  justifyContent: 'center',
  margin: 20,
};

const generateCanvas = (
  canvasRef: React.RefObject<HTMLCanvasElement>,
  idx: number
) => (
  <canvas
    style={{ margin: 10 }}
    key={`key-${idx}`}
    ref={canvasRef}
    width={SIZE}
    height={SIZE}
  />
);

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
    (async () => {
      const exampleImages: string[] = await makeRequest(
        `example-images/${this.props.isNormal}/${this.props.N}`,
        JSON.parse
      );
      for (let i = 0; i < this.props.N; i++) {
        const canvas = this.canvasRefs[i].current!;
        writeBase64ToCanvas(canvas, exampleImages[i], SIZE);
      }
    })();
  };

  render() {
    return (
      <div style={canvasLayoutStyle}>{this.canvasRefs.map(generateCanvas)}</div>
    );
  }
}

export default ExampleImages;

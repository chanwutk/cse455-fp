import React from 'react';
import { writeBase64ToCanvas } from '../utils';
import IFFValid from './IFFValid';

const SIZE = 300;

interface MaskedImagesProps {
  masks: null | string[];
}

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

class MaskedImages extends React.Component<MaskedImagesProps> {
  canvasRefs: React.RefObject<HTMLCanvasElement>[];

  constructor(props: MaskedImagesProps) {
    super(props);
    this.canvasRefs = [];
  }

  componentDidUpdate = () => {
    (async () => {
      if (this.props.masks !== null) {
        for (let i = 0; i < this.props.masks.length; i++) {
          const canvas = this.canvasRefs[i].current!;
          writeBase64ToCanvas(canvas, this.props.masks[i], SIZE);
        }
      }
    })();
  };

  render() {
    this.canvasRefs = [];
    if (this.props.masks !== null) {
      for (let i = 0; i < this.props.masks.length; i++) {
        this.canvasRefs.push(React.createRef());
      }
    }

    return (
      <IFFValid validate={this.props.masks}>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {((masks: string[] | null) => {
            console.log(masks);
            const output = [];
            if (masks !== null) {
              for (let i = 0; i < masks.length; i += 3) {
                const row = [];
                for (let j = 0; j < 3 && j + i < masks.length; j++) {
                  row.push(generateCanvas(this.canvasRefs[i + j], i + j));
                }
                output.push(
                  <div
                    key={i}
                    style={{ display: 'flex', flexDirection: 'row' }}
                  >
                    {row}
                  </div>
                );
              }
            }
            return output;
          })(this.props.masks!)}
        </div>
      </IFFValid>
    );
  }
}

export default MaskedImages;

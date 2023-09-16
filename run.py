import argparse
import os
import cv2
import numpy as np
from capture import MotionCapture


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config.json")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, args.config)

    capture = MotionCapture(config_path)
    capture.start(mode='multi-process')
    
    while True:
        data = capture.read()
        print(data is not None)
        if data:
            cv2.imshow("manager", np.zeros([160,90]))
        if cv2.waitKey(1) == 27:
            break

    capture.end()


if __name__ == '__main__':
    main()

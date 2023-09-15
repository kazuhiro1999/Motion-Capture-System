import argparse
import os
import cv2
import numpy as np
from controller import MotionCaptureController


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config.json")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, args.config)

    controller = MotionCaptureController()
    controller.initialize(config_path=config_path)
    
    #controller.debug = True
    controller.start_capture(mode='multi-process')
    
    while True:
        data = controller.get_data()
        if data:
            cv2.imshow(data['TimeStamp'], np.zeros([64,64]))
        if cv2.waitKey(1) == 27:
            break

    controller.end_capture()


if __name__ == '__main__':
    main()

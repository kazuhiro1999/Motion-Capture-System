import argparse
import os
from controller import MotionCaptureController


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config1.json")
    parser.add_argument('--host', default="127.0.0.1")
    parser.add_argument('--port', default=50000)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, args.config)

    controller = MotionCaptureController()
    controller.initialize(config_path=config_path, udp_host=args.host, udp_port=args.port)
    
    #controller.debug = True
    controller.start_capture(mode='multi-process')
    
    input()
    controller.end_capture()


if __name__ == '__main__':
    main()

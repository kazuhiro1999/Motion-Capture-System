import argparse
from controller import MotionCaptureController


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config.json")
    parser.add_argument('--host', default="127.0.0.1")
    parser.add_argument('--port', default=50000)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    controller = MotionCaptureController()
    controller.initialize(config_path=args.config, udp_host=args.host, udp_port=args.port)

    controller.start_capture()

    while True:
        if input() == 'q':
            break
    
    controller.end_capture()


if __name__ == '__main__':
    main()

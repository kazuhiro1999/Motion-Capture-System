import cv2
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Event

from utils import TimeUtil
from camera import USBCamera
from pose2d import MediapipePose
from pose3d import recover_pose_3d
from network import UDPClient
from data import to_dict
from multiprocess import process_start
from visalization import draw_keypoints3d


class MotionCaptureController:
    def __init__(self):
        self.config = None
        self.is_playing = False
        self.host = '127.0.0.1'
        self.port = 50000
        self.debug = False

    def initialize(self, config_path, udp_host, udp_port):
        self.config = self.load_config(config_path)
        self.host = udp_host
        self.port = udp_port
        print(f"Capture initialized with config: {config_path} and UDP Port: {udp_port}")

    def start_capture(self, mode='default'):
        if self.is_playing:
            return 'Capture has already started'
        
        if self.config is None or self.port is None:
            return 'Capture is not initialized yet. Please call initialize() before start.'

        self.is_playing = True
        self.mode = mode
        if mode == 'default':
            print("capture started at main thread. Press ESC to end capture.")
            self.capture_process()
        elif mode == 'multi-thread':
            self.capture_thread = threading.Thread(target=self.capture_process)
            self.capture_thread.start()
            print("capture thread started")
        elif mode == 'multi-process':
            self.cancel_event = Event()
            self.proc = Process(target=process_start, args=(self.config, self.host, self.port, self.cancel_event))
            self.proc.start()            
            print("capture process started")
        else:
            raise Exception(f"unknown capture mode : {mode}")
        return "Success to start capture"

    def end_capture(self):
        if not self.is_playing:
            return "Capture has not been started"
        
        self.is_playing = False
        if self.mode == 'multi-thread':
            self.capture_thread.join()
        elif self.mode == 'multi-process':
            self.cancel_event.set()
            self.proc.join()
            self.proc.terminate()
        else:
            pass
        return "Capture ended"

    def load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Failed to load config: {e}")
            return None

    def capture_process(self):
        # Initialize Motion Capture
        cameras = [USBCamera(camera_config) for camera_config in self.config['cameras']]
        for camera in cameras:
            camera.open()

        pose_estimators = [MediapipePose() for _ in self.config['cameras']]

        udp_client = UDPClient(host=self.host, port=self.port)
        udp_client.open()

        keypoints2d_list = []
        proj_matrices = []

        t = TimeUtil.get_unixtime()
        print("capture start")
        # Main loop
        with ThreadPoolExecutor(max_workers=4) as executor:

            while self.is_playing:
                timestamp = t
                # send 2d pose estimation
                t = TimeUtil.get_unixtime()
                futures = []
                for camera, pose_estimator in zip(cameras, pose_estimators):
                    frame = camera.get_image()
                    if frame is not None:
                        cv2.imshow(camera.name, cv2.resize(frame, dsize=(640,360)))
                        future = executor.submit(pose_estimator.process, frame)
                        futures.append(future)

                # 3d pose estimation 
                future_keypoints3d = executor.submit(recover_pose_3d, proj_matrices, keypoints2d_list)

                keypoints3d = future_keypoints3d.result()
                data = to_dict(timestamp, keypoints3d, MediapipePose.KEYPOINT_DICT, MediapipePose.Type)                
                ret = udp_client.send(data)

                if self.debug:
                    draw_keypoints3d(keypoints3d, pose_estimator.KINEMATIC_TREE)  # for visualization


                # get 2d pose estimation results
                keypoints2d_list = []
                proj_matrices = []

                for camera, pose_estimator, future in zip(cameras, pose_estimators, futures):
                    keypoints2d = future.result()
                    if keypoints2d is not None:
                        proj_matrix = camera.camera_setting.proj_matrix
                        proj_matrices.append(proj_matrix)
                        keypoints2d_list.append(keypoints2d)


                if cv2.waitKey(1) == 27:  # Press ESC to exit
                    break

        # End Motion Caapture
        for camera in cameras:
            camera.close()

        udp_client.close()

        cv2.destroyAllWindows()
        return

import cv2
import numpy as np
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from camera import USBCamera
from pose3d import recover_pose_3d
from pose2d import MediapipePose
from network import UDPClient
from utils import TimeUtil
from visalization import draw_keypoints3d


class MotionCaptureController:
    def __init__(self):
        self.config = None
        self.is_playing = False
        self.host = '127.0.0.1'
        self.port = 50000

    def initialize(self, config_path, udp_host, udp_port):
        self.config = self.load_config(config_path)
        self.host = udp_host
        self.port = udp_port
        print(f"Capture initialized with config: {config_path} and UDP Port: {udp_port}")

    def start_capture(self):
        if self.is_playing:
            print('Capture has already started')
            return
        if self.config is None or self.port is None:
            print('Capture is not initialized yet. Please call initialize() before start.')
            return
        self.is_playing = True
        self.capture_thread = threading.Thread(target=self.capture_process)
        self.capture_thread.start()
        print(f"Capture started")

    def end_capture(self):
        if not self.is_playing:
            print("Capture has not been started")
            return
        self.is_playing = False
        self.capture_thread.join()
        print("Capture ended")

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
                if keypoints3d is not None:
                    data = {"Type":MediapipePose.Type, "TimeStamp": timestamp, "Bones":[]}
                    keys = MediapipePose.KEYPOINT_DICT
                    for key in keys:
                        bone = {
                            "Name": key,
                            "Position":{
                                "x": float(keypoints3d[keys[key],0]),
                                "y": float(keypoints3d[keys[key],1]),
                                "z": float(keypoints3d[keys[key],2]),
                            }
                        }
                        data['Bones'].append(bone)                
                    ret = udp_client.send(data)
                #draw_keypoints3d(keypoints3d, pose_estimator.KINEMATIC_TREE)  # for visualization


                # get 2d pose estimation results
                keypoints2d_list = []
                proj_matrices = []

                for camera, pose_estimator, future in zip(cameras, pose_estimators, futures):
                    keypoints2d = future.result()
                    if keypoints2d is not None:
                        proj_matrix = camera.camera_setting.get_projection_matrix()
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

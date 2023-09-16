from queue import Empty
import cv2
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Event, Queue

from utils import TimeUtil
from camera import USBCamera
from pose2d import MediapipePose
from pose3d import recover_pose_3d
from network import UDPClient
from data import to_dict
from multiprocess import capture_process
from visalization import draw_keypoints3d


class Status:
    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    CAPTURING = "capturing"

class MotionCapture:

    def __init__(self, config_path=None):
        """Initialize the MotionCapture."""
        self.config = None
        self.is_playing = False
        self.queue = Queue()
        self.debug = False
        self.status = Status.UNINITIALIZED
        
        if config_path:
            self.load_config(config_path)

    def read(self, timeout=1.0):
        """
        Fetch data from the queue if available.

        Args:
            timeout (float): Max time in seconds to wait for data.

        Returns:
            dict or None: Retrieved data or None if not available.
        """
        if self.status != Status.CAPTURING:
            return None
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None

    def start(self, mode='multi-process'):
        """
        Start the motion capture process.

        Args:
            mode (str): Mode to run the capture process. Options are 'default', 'multi-thread', and 'multi-process'.

        Returns:
            str: Status message.
        """
        if self.status == Status.CAPTURING:
            return 'Capture has already started'
        
        if self.status == Status.UNINITIALIZED:
            return 'Capture is not initialized yet.'

        self.is_playing = True
        self.mode = mode

        if mode == 'default':
            print("capture started at main thread. Press ESC to end capture.")
            self.capture_thread()
        elif mode == 'multi-thread':
            self.capture_thread = threading.Thread(target=self.capture_thread)
            self.capture_thread.start()
            print("capture thread started")
        elif mode == 'multi-process':
            self.cancel_event = Event()
            self.proc = Process(target=capture_process, args=(self.config, self.queue, self.cancel_event))
            self.proc.start()            
            print("capture process started")
        else:
            raise Exception(f"Unknown capture mode : {mode}")
        
        self.status = Status.CAPTURING
        return "Success to start capture"

    def end(self):
        """
        End the motion capture process.

        Returns:
            str: Status message.
        """
        if self.status != Status.CAPTURING:
            return "Capture has not been started"
        
        self.is_playing = False

        if self.mode == 'multi-thread':
            self.capture_thread.join()
        elif self.mode == 'multi-process':
            self.cancel_event.set()
            self.proc.join()
            self.proc.terminate()
        
        self.status = Status.INITIALIZED
        return "Capture ended"

    def load_config(self, config_path):
        """
        Load configuration from a file.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            dict or None: Loaded configuration or None if failed.
        """
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"Failed to load config: {e}")
            return None
        
        self.status = Status.INITIALIZED
        return self.config
    
    def calibrate(self):
        """
        Calibrate the motion capture system.
        
        This function is a placeholder for future implementation.
        """
        pass


    def capture_thread(self):
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
                self.queue.put(data)            
                #ret = udp_client.send(data)

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

import cv2
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Event, Queue
from queue import Empty

from utils import TimeUtil
from camera import CameraSetting, USBCamera
from pose2d import MediapipePose
from pose3d import recover_pose_3d
from network import UDPClient
from data import to_dict
from multiprocess import camera_process, capture_process
from space_calibration import SpaceCalibrator
from visalization import draw_keypoints3d


class Status:
    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    CAPTURING = "capturing"
    CALIBRATING = "calibrating"


class MotionCapture:

    def __init__(self, config_path=None):
        """Initialize the MotionCapture."""
        self.config_path = config_path
        self.config = None
        self.is_playing = False
        self.queue = Queue()
        self.debug = False
        self.calibrator = SpaceCalibrator()
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
            return False, 'Capture has already started'
        
        if self.status == Status.UNINITIALIZED:
            return False, 'Capture is not initialized yet.'
        
        if self.status == Status.CALIBRATING:
            return False, 'Cannot start capture. Calibration is in progress.'

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
        return True, "Success to start capture"

    def end(self):
        """
        End the motion capture process.

        Returns:
            str: Status message.
        """
        if self.status != Status.CAPTURING:
            return False, "Capture has not been started"
                
        self.is_playing = False

        if self.mode == 'multi-thread':
            self.capture_thread.join()
        elif self.mode == 'multi-process':
            self.cancel_event.set()
            self.proc.join()
            self.proc.terminate()
        
        self.status = Status.INITIALIZED
        return True, "Capture ended"

    def load_config(self, config_path):
        """
        Load configuration from a file.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            bool: True if config loaded successfully, False otherwise.
            str: Message indicating the result.
        """
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.config_path = config_path
            self.status = Status.INITIALIZED
            return True, "Config loaded successfully"
        except Exception as e:
            return False, f"Failed to load config: {e}"
    
    def calibrate(self, reference_height=1.6):
        """
        Calibrate the motion capture system.
        """
        if self.status != Status.INITIALIZED:
            return False, "Cannot calibrate. Capture is not initialized."
                
        self.status = Status.CALIBRATING  # キャリブレーション開始時にステータスを更新

        # Initialize 
        processes = []
        queues = []
        events = []
        camera_settings = []
        cancel_event = Event()
        try:
            for camera_config in self.config['cameras']:
                queue = Queue()
                event = Event()
                process = Process(target=camera_process, args=(camera_config, event, queue, cancel_event))

                processes.append(process)
                queues.append(queue)
                events.append(event)

                process.start()

                camera_setting = CameraSetting(camera_config['setting_path'])
                camera_settings.append(camera_setting)

            if len(camera_settings) == 0:
                raise ValueError("No camera settings provided.")
            
            # Start calibration
            self.calibrator.start_calibration(camera_settings)

            # Main loop
            while not self.calibrator.is_sampled():
                # Send signal to start processing
                for event in events:
                    event.set()

                timestamp = TimeUtil.get_unixtime()

                # Collect results from each process
                keypoints2d_list = []

                for queue in queues:
                    try:
                        frame, keypoints2d = queue.get(timeout=1.0)
                    except Empty:
                        continue
                    if keypoints2d is not None:
                        keypoints2d_list.append(keypoints2d)

                # add sample
                self.calibrator.add_sample(timestamp, keypoints2d_list)

            # execute camera calibration
            camera_settings, keypoints3d_list = self.calibrator.calibrate_cameras(reference_height=reference_height)

            # save calibration result
            for camera_setting in camera_settings:
                camera_setting.save()

            # reload config
            self.load_config(self.config_path)
            self.status = Status.INITIALIZED  # キャリブレーション終了後にステータスを更新
            print("calibration ended")
            
            return True, "Calibration successful"

        except Exception as e:
            return False, f"Error occurred during calibration: {e}"

        finally:
            # End subprocess
            cancel_event.set()
            for process in processes:
                process.join()
                process.terminate()

            cv2.destroyAllWindows() 


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

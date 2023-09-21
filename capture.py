import time
import json
from multiprocessing import Process, Event, Queue
from queue import Empty

from camera import CameraSetting
from space_calibration import SpaceCalibrator
from multiprocess import capture_process
from utils import TimeUtil


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
            return True, f"Config loaded successfully: {config_path}"
        except FileNotFoundError:
            return False, f"Config file not found: {e}"
        except json.JSONDecodeError:
            return False, f"Failed to parse config: {e}"
        except Exception as e:
            return False, f"Unknown error occurred while loading config: {e}"

    def read(self, timeout=1.0):
        """
        Fetch data from the queue if available.

        Args:
            timeout (float): Max time in seconds to wait for data.

        Returns:
            dict or None: Retrieved data or None if not available.
        """
        if self.status != Status.CAPTURING and self.status != Status.CALIBRATING:
            return None, None, None
        try:
            timestamp, keypoints2d_list, keypoints3d = self.queue.get(timeout=timeout)   
            timestamp = timestamp + TimeUtil.Offset
            return timestamp, keypoints2d_list, keypoints3d
        except Empty:
            return None, None, None

    def start(self):
        """
        Start the motion capture process.

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
        self.cancel_event = Event()
        self.proc = Process(target=capture_process, args=(self.config, self.queue, self.cancel_event))
        self.proc.start()  
        self.status = Status.CAPTURING   

        print("capture process started")        
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
        self.cancel_event.set()
        self.proc.join(timeout=1.0)
        self.proc.terminate()    
        self.status = Status.INITIALIZED

        print("capture process ended")        
        return True, "Capture ended"
        
    def init_calibration(self):
        """
        Init calibrator and start the motion capture process.

        Returns:
            str: Status message.
        """
        if self.status == Status.CAPTURING:
            return False, 'Capture has already started.'
        
        if self.status == Status.UNINITIALIZED:
            return False, 'Capture is not initialized yet.'
        
        if self.status == Status.CALIBRATING:
            return False, 'Calibration is in progress.'

        self.is_playing = True
        self.cancel_event = Event()
        self.proc = Process(target=capture_process, args=(self.config, self.queue, self.cancel_event, 2))
        self.proc.start()  
        self.status = Status.CAPTURING   

        print("calibration initialized")        
        return True, "Start capturing for calibration."
    
    def start_calibration(self, reference_height=1.6):
        """
        Calibrate the motion capture system.
        """
        if self.status != Status.CAPTURING:
            return False, "Cannot start calibration. Capture is not started."
                
        camera_settings = [CameraSetting(camera_config['setting_path']) for camera_config in self.config['cameras']]
        self.calibrator.initialize(camera_settings)
        self.status = Status.CALIBRATING  # キャリブレーション開始時にステータスを更新
        
        try:
            print("sampling...")
            while not self.calibrator.is_sampled():
                timestamp, keypoints2d_list, keypoints3d = self.read()
                if timestamp:
                    self.calibrator.add_sample(timestamp, keypoints2d_list)
                print(len(self.calibrator.samples))
                time.sleep(0.01)

            # execute camera calibration
            camera_settings, keypoints3d_list = self.calibrator.calibrate_cameras(reference_height=reference_height)

            # save calibration result
            for camera_setting in camera_settings:
                camera_setting.save()

            # reload config
            self.load_config(self.config_path)
            self.status = Status.CAPTURING  # キャリブレーション終了後にステータスを更新
            print("calibration ended")
            
            return True, "Calibration successful"

        except Exception as e:
            return False, f"Error occurred during calibration: {e}"

    def correct_calibration_with_hmd_trajectory(self, trajectory_data):
        try:
            self.calibrator.correct_calibration_with_hmd_trajectory(trajectory_data)
            # reload config
            self.load_config(self.config_path)
            return True, "Room calibration with HMD successful"
        except Exception as e:
            return False, f"Error occurred during room calibration: {e}"

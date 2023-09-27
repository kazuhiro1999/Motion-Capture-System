import cv2
import numpy as np
import json


class USBCamera:
    def __init__(self, config):
        self.name = config['name']
        self.device_id = config['device']
        self.is_active = False
        self.camera_setting = CameraSetting(config['setting_path'])

    def open(self):
        if self.is_active:
            print(f'{self.name} is already open.')
            return self.is_active
        self.cap = cv2.VideoCapture(self.device_id)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            self.is_active = False
            return self.is_active
        self.is_active = True
        return self.is_active

    def get_image(self):
        if not self.is_active:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        if self.camera_setting.image_width > 0 and self.camera_setting.image_height > 0:
            frame = cv2.resize(frame, dsize=(self.camera_setting.image_width, self.camera_setting.image_height))
        if self.camera_setting.distortion_coefficients is not None:  # 歪み補正
            frame = cv2.undistort(frame, self.camera_setting.intrinsic_matrix, self.camera_setting.distortion_coefficients)
        return frame

    def close(self):
        if self.cap is not None:
            self.cap.release()
        self.is_active = False
        return True



class CameraSetting:
    def __init__(self, setting_path=None):
        self.setting_path = setting_path
        if setting_path is not None:
            self.load(setting_path)
        else:
            self.image_width = 0
            self.image_height = 0
            self.intrinsic_matrix = np.eye(3)
            self.distortion_coefficients = np.zeros(5)
            self.extrinsic_matrix = np.eye(4)
            self.proj_matrix = self.get_projection_matrix()

    def load(self, setting_path):
        with open(setting_path, 'r') as f:
            config = json.load(f)
            self.image_width = config.get('image_width', 0)
            self.image_height = config.get('image_height', 0)
            self.intrinsic_matrix = np.array(config.get('intrinsic_matrix', np.eye(3)))
            self.distortion_coefficients = np.array(config.get('distortion_coefficients', np.zeros(5)))
            self.extrinsic_matrix = np.array(config.get('extrinsic_matrix', np.eye(4)))
            self.proj_matrix = self.get_projection_matrix()
        self.setting_path = setting_path

    def save(self):
        if self.setting_path is None:
            return
        config = {
            'image_width': self.image_width,
            'image_height': self.image_height,
            'intrinsic_matrix': self.intrinsic_matrix.tolist(),
            'extrinsic_matrix': self.extrinsic_matrix.tolist(),
            'distortion_coefficients': self.distortion_coefficients.tolist()
        }
        with open(self.setting_path, 'w') as f:
            json.dump(config, f, indent=4)

    def get_projection_matrix(self):
        return self.intrinsic_matrix @ self.extrinsic_matrix


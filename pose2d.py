import numpy as np
import mediapipe as mp
from abc import ABC, abstractmethod


class PoseEstimator(ABC):

    @abstractmethod
    def process(self, image):
        pass


class MediapipePose(PoseEstimator):
    Type = 'MediapipePose'
    KEYPOINT_DICT = {
        'nose': 0,
        'left_inner_eye': 1,
        'left_eye': 2,
        'left_outer_eye': 3,
        'right_inner_eye': 4,
        'right_eye': 5,
        'right_outer_eye': 6,
        'left_ear': 7,
        'right_ear':8,
        'left_mouth': 9,
        'right_mouth': 10,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_outer_hand': 17,
        'right_outer_hand': 18,
        'left_hand_tip': 19,
        'right_hand_tip': 20,
        'left_inner_hand': 21,
        'right_inner_hand': 22,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
        'left_heel': 29,
        'right_heel': 30,
        'left_toe': 31,
        'right_toe': 32
    }

    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.5,
            )

    def process(self, image):
        image_height, image_width = image.shape[:2]
        results = self.pose.process(image)
        if results.pose_landmarks is not None:
            keypoints2d = self.landmarks_to_keypoints2d(results.pose_landmarks, (image_width, image_height))
        else:
            keypoints2d = None
        return keypoints2d

    def landmarks_to_keypoints2d(self, landmarks, image_shape):
        image_width, image_height = image_shape
        keypoints2d = []
        for landmark in landmarks.landmark:
            x = landmark.x * image_width
            y = landmark.y * image_height
            keypoints2d.append([x, y, landmark.visibility])
        return np.array(keypoints2d)


import numpy as np
from pose2d import MediapipePose

KEYPOINT_DICT = MediapipePose.KEYPOINT_DICT


def determine_scale(keypoints3d_list, Height=1.6):
    head = keypoints3d_list[:, KEYPOINT_DICT['nose']]
    l_foot = keypoints3d_list[:, KEYPOINT_DICT['left_ankle']]
    r_foot = keypoints3d_list[:, KEYPOINT_DICT['right_ankle']]
    m_foot = (l_foot + r_foot) / 2
    height = np.linalg.norm(head - m_foot, axis=-1).mean()
    scale = Height / height
    return scale

def determine_center_position(keypoints3d_list):
    # keypoints3d_list : (n_frames, n_joints, 3)
    l_foot = keypoints3d_list[:, KEYPOINT_DICT['left_ankle']]
    r_foot = keypoints3d_list[:, KEYPOINT_DICT['right_ankle']]
    l_toe = keypoints3d_list[:, KEYPOINT_DICT['left_toe']]
    r_toe = keypoints3d_list[:, KEYPOINT_DICT['right_toe']]
    m_foot = (l_foot + r_foot + l_toe + r_toe) / 4
    center_position = m_foot.mean(axis=0)
    return center_position

def determine_forward_vector(keypoints3d_list):
    # 初めの数フレームの上半身の前方
    l_shoulder = keypoints3d_list[:, KEYPOINT_DICT['left_shoulder']]
    r_shoulder = keypoints3d_list[:, KEYPOINT_DICT['right_shoulder']]
    l_hips = keypoints3d_list[:, KEYPOINT_DICT['left_hip']]
    r_hips = keypoints3d_list[:, KEYPOINT_DICT['right_hip']]    
    m_hips = (l_hips + r_hips) / 2
    # 両肩および腰のクロス積
    forward_vector = np.cross(l_shoulder - m_hips, r_shoulder - m_hips).mean(axis=0)
    forward = forward_vector / np.linalg.norm(forward_vector)
    return forward

def determine_upward_vector(keypoints3d_list):
    # 初めの数フレームの直立姿勢方向
    l_foot = keypoints3d_list[:, KEYPOINT_DICT['left_ankle']]
    r_foot = keypoints3d_list[:, KEYPOINT_DICT['right_ankle']]
    l_toe = keypoints3d_list[:, KEYPOINT_DICT['left_toe']]
    r_toe = keypoints3d_list[:, KEYPOINT_DICT['right_toe']]
    m_foot = (l_foot + r_foot + l_toe + r_toe) / 4 
    l_shoulder = keypoints3d_list[:, KEYPOINT_DICT['left_shoulder']]
    r_shoulder = keypoints3d_list[:, KEYPOINT_DICT['right_shoulder']]
    m_shoulder = (l_shoulder + r_shoulder) / 2
    # 足元から首へのベクトル
    up_vector = (m_shoulder - m_foot).mean(axis=0)
    up = up_vector / np.linalg.norm(up_vector)
    return up

def rotation_matrix_from_vectors(forward_vector, up_vector):
    # Normalize the input vectors
    forward_vector = forward_vector / np.linalg.norm(forward_vector)
    up_vector = up_vector / np.linalg.norm(up_vector)
    
    # Compute the right vector using cross product
    right_vector = np.cross(up_vector, forward_vector)
    right_vector /= np.linalg.norm(right_vector)
    
    # Compute the actual up vector using cross product
    actual_up_vector = np.cross(forward_vector, right_vector)
    
    # Construct the rotation matrix
    R = np.column_stack((right_vector, actual_up_vector, forward_vector))
    return R

# 同一平面にある点群に対する直行ベクトルを計算
def normal_vector(points3d):
    # points3d : (n_points, 3)   
    centroid = np.mean(points3d, axis=0)
    points = points3d - centroid
    covariance_matrix = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    normal_vector = eigenvectors[:, np.argmin(eigenvalues)]
    return normal_vector
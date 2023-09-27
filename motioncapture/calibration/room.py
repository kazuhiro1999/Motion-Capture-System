import numpy as np
from ..pose2d import MediapipePose

KEYPOINT_DICT = MediapipePose.KEYPOINT_DICT


def determine_scale(keypoints3d_list, num_frames=10, skip_frames=5, height=1.0):
    """
    Determines the scale factor based on the provided 3D keypoints.
    
    Args:
    - keypoints3d_list (numpy.ndarray): 3D keypoints of shape (n_frames, n_joints, 3).
    - num_frames (int): Number of frames to consider for computing the scale.
    - height (float): The expected height to scale to.
    
    Returns:
    - scale (float): The computed scale factor.
    """
    subset = keypoints3d_list[skip_frames:skip_frames+num_frames]
    
    head = subset[:, KEYPOINT_DICT['nose']]
    l_foot = subset[:, KEYPOINT_DICT['left_ankle']]
    r_foot = subset[:, KEYPOINT_DICT['right_ankle']]
    m_foot = (l_foot + r_foot) / 2
    computed_height = np.linalg.norm(head - m_foot, axis=-1).mean()
    scale = height / computed_height
    return scale


def determine_center_position(keypoints3d_list, num_frames=10, skip_frames=5):
    """
    Determines the center position based on the provided 3D keypoints.
    
    Args:
    - keypoints3d_list (numpy.ndarray): 3D keypoints of shape (n_frames, n_joints, 3).
    - num_frames (int): Number of frames to consider for computing the center position.
    
    Returns:
    - center_position (numpy.ndarray): The computed center position.
    """
    subset = keypoints3d_list[skip_frames:skip_frames+num_frames]
    
    l_foot = subset[:, KEYPOINT_DICT['left_ankle']]
    r_foot = subset[:, KEYPOINT_DICT['right_ankle']]
    l_toe = subset[:, KEYPOINT_DICT['left_toe']]
    r_toe = subset[:, KEYPOINT_DICT['right_toe']]
    m_foot = (l_foot + r_foot + l_toe + r_toe) / 4
    center_position = m_foot.mean(axis=0)
    return center_position


def determine_forward_vector(keypoints3d_list, num_frames=10, skip_frames=5):
    """
    Determines the forward direction based on the provided 3D keypoints.
    
    Args:
    - keypoints3d_list (numpy.ndarray): 3D keypoints of shape (n_frames, n_joints, 3).
    - num_frames (int): Number of frames to consider for computing the forward direction.
    
    Returns:
    - forward (numpy.ndarray): The computed forward direction vector.
    """
    subset = keypoints3d_list[skip_frames:skip_frames+num_frames]
    
    l_shoulder = subset[:, KEYPOINT_DICT['left_shoulder']]
    r_shoulder = subset[:, KEYPOINT_DICT['right_shoulder']]
    l_hips = subset[:, KEYPOINT_DICT['left_hip']]
    r_hips = subset[:, KEYPOINT_DICT['right_hip']]    
    m_hips = (l_hips + r_hips) / 2
    
    forward_vector = np.cross(l_shoulder - m_hips, r_shoulder - m_hips).mean(axis=0)
    forward = forward_vector / np.linalg.norm(forward_vector)
    return forward


def determine_upward_vector(keypoints3d_list, num_frames=10, skip_frames=5):
    # keypoints3d_list : (n_frames, n_joints, 3)   
    
    # Compute the normal vector based on the 'nose' keypoint distribution
    nose_points = keypoints3d_list[:, KEYPOINT_DICT['nose']]
    normal = normal_vector(nose_points)
    
    # Use the specified number of frames to estimate the upward direction
    subset = keypoints3d_list[skip_frames:skip_frames+num_frames]
    
    l_foot = subset[:, KEYPOINT_DICT['left_ankle']]
    r_foot = subset[:, KEYPOINT_DICT['right_ankle']]
    l_toe = subset[:, KEYPOINT_DICT['left_toe']]
    r_toe = subset[:, KEYPOINT_DICT['right_toe']]
    
    # Compute the average position of feet and toes for the specified frames
    avg_foot_pos = (l_foot + r_foot + l_toe + r_toe) / 4 
    
    l_shoulder = subset[:, KEYPOINT_DICT['left_shoulder']]
    r_shoulder = subset[:, KEYPOINT_DICT['right_shoulder']]
    
    # Compute the average position of shoulders for the specified frames
    avg_shoulder_pos = (l_shoulder + r_shoulder) / 2
    
    # The expected upward direction is from feet towards shoulders
    expected_up_vector = (avg_shoulder_pos - avg_foot_pos).mean(axis=0)
    expected_upward = expected_up_vector / np.linalg.norm(expected_up_vector)
    
    # Ensure the computed normal vector aligns with the expected upward direction
    if np.dot(normal, expected_upward) < 0:
        up = -normal
    else:
        up = normal
    return up


def rotation_matrix_from_vectors(forward_vector, up_vector):
    # Normalize the input vectors
    forward_vector = forward_vector / np.linalg.norm(forward_vector)
    up_vector = up_vector / np.linalg.norm(up_vector)
    
    # Compute the right vector using cross product
    right_vector = np.cross(up_vector, forward_vector)
    right_vector /= np.linalg.norm(right_vector)
    
    # Compute the actual up vector using cross product
    forward_vector = np.cross(right_vector, up_vector)
    
    # Construct the rotation matrix
    R = np.column_stack((right_vector, up_vector, forward_vector))
    
    return R


# 平面の直行ベクトルを計算
def normal_vector(points3d):
    # points3d : (n_points, 3)   
    centroid = np.mean(points3d, axis=0)
    points = points3d - centroid
    covariance_matrix = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    normal_vector = eigenvectors[:, np.argmin(eigenvalues)]
    return normal_vector
import json
import numpy as np
from calibration.room import *
from pose2d import MediapipePose

KEYPOINT_DICT = MediapipePose.KEYPOINT_DICT


def load_json_data(file_path):
    """Load data from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_hmd_trajectory(trajectory_data, result_data):
    """Extract synchronized HMD trajectory based on timestamps."""
    timestamps = np.array([transform['timestamp'] for transform in trajectory_data['transforms']])
    hmd_trajectory = np.array([[transform['position']['x'], transform['position']['y'], transform['position']['z']] for transform in trajectory_data['transforms']])
    
    t_list = np.array(result_data['timestamps']) - 200  # Adjusting for time offset
    
    # Use NumPy's searchsorted for efficient timestamp matching
    indices = np.searchsorted(timestamps, t_list)
    hmd_trajectory_extracted = hmd_trajectory[indices]
    
    return hmd_trajectory_extracted

def load_trajectory(folder_path):
    """Load and synchronize HMD trajectory data."""
    trajectory_data = load_json_data(f"{folder_path}/trajectory.json")
    result_data = load_json_data(f"{folder_path}/result.json")
    
    return extract_hmd_trajectory(trajectory_data, result_data)


def best_fit_transform_kabsch(A, B, translation=True):
    """
    Computes the best-fit transform that aligns points in A with points in B using Kabsch algorithm.
    
    Args:
    - A (numpy.ndarray): Source points of shape (n, 3), where n is the number of points.
    - B (numpy.ndarray): Target points of shape (n, 3).
    
    Returns:
    - R (numpy.ndarray): Optimal rotation matrix of shape (3, 3).
    - T (numpy.ndarray): Optimal translation vector of shape (3,).
    """
    # Center the datasets
    #centroid_A = np.mean(A, axis=0)
    #centroid_B = np.mean(B, axis=0)
    centroid_A = np.array([0, np.mean(A[:, 1]), 0])
    centroid_B = np.array([0, np.mean(B[:, 1]), 0])
    
    A_centered = A - centroid_A
    B_centered = B - centroid_B
    
    # Compute the covariance matrix
    covariance_matrix = np.dot(A_centered.T, B_centered)
    
    # Compute the Singular Value Decomposition
    U, _, Vt = np.linalg.svd(covariance_matrix)
    
    # Ensure a right-handed coordinate system
    d = (np.linalg.det(U) * np.linalg.det(Vt)) < 0.0
    
    if d:
        Vt[-1, :] *= -1
    
    # Compute the rotation matrix
    R = np.dot(Vt.T, U.T)
    
    # Compute the translation
    if translation:
        T = centroid_B - np.dot(centroid_A, R)
    else:
        T = np.zeros(3)
    
    return R, T


def room_calibration_with_trajectry(camera_params, keypoints3d_list, trajectry, K):

    height = trajectry[:,1].mean()
    s0 = determine_scale(keypoints3d_list, height=height)
    p0 = determine_center_position(keypoints3d_list)
    forward = determine_forward_vector(keypoints3d_list)
    up = determine_upward_vector(keypoints3d_list)
    R0 = rotation_matrix_from_vectors(forward, up)
    t0 = -R0 @ (p0 * s0) 
    
    trajectry_pred = keypoints3d_list[:, KEYPOINT_DICT['nose'], :]
    trajectry_pred = (s0 * trajectry_pred) @ R0.T + t0
    
    Rs, Ts = best_fit_transform_kabsch(trajectry_pred, trajectry, translation=False)
    R_final = Rs @ R0
    t_final = t0 @ Rs.T + Ts

    # カメラパラメータ計算
    new_camera_params = {}
    for i in range(len(camera_params)):
        R = R_final @ camera_params[i]['R']
        t = R_final @ (s0 * camera_params[i]['t']) + t_final.reshape([3,1])
        Rc = R.T
        tc = -R.T @ t
        proj_matrix = K.dot(np.concatenate([Rc,tc], axis=-1))
        new_camera_params[i] = {
            'R': R,
            't': t,
            'proj_matrix': proj_matrix
        }
    return new_camera_params
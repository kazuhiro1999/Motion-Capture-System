import numpy as np
from calibration.room import *
from pose2d import MediapipePose

KEYPOINT_DICT = MediapipePose.KEYPOINT_DICT


def room_calibration_with_trajectry(camera_params, keypoints3d_list, head_trajectry, K):

    s0 = determine_scale(keypoints3d_list[5:25], Height=1.6)
    p0 = determine_center_position(keypoints3d_list[5:25])
    forward = determine_forward_vector(keypoints3d_list[5:25])
    up = determine_upward_vector(keypoints3d_list[5:25])
    R0 = rotation_matrix_from_vectors(forward, up)
    t0 = -R0 @ (p0 * s0) 
    
    head_trajectry_pred = keypoints3d_list[:, KEYPOINT_DICT['nose'], :]
    head_trajectry_pred = (s0 * head_trajectry_pred) @ R0.T + t0
    
    Rs, Ts = best_fit_transform(head_trajectry_pred, head_trajectry)
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

def moving_average(data, window_size):
    """
    Compute the moving average of the data with the specified window size.
    """
    cumsum = np.cumsum(data, axis=0)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

def compute_directions(trajectory):
    """
    Compute the directions of a trajectory.
    """
    directions = np.diff(trajectory, axis=0)
    # Normalize the directions
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    return directions

def compute_rotation(A, B):
    """
    Compute the rotation matrix that aligns A with B.
    """
    # Compute the cross product of directions to get the rotation axis
    cross_product = np.cross(A, B)
    # Compute the dot product to get the cosine of the angle
    dot_product = np.sum(A * B, axis=1)
    # Compute the rotation angle
    angles = np.arctan2(np.linalg.norm(cross_product, axis=1), dot_product)
    # Average the rotation angles
    mean_angle = np.mean(angles)
    # Rotation axis
    mean_axis = np.mean(cross_product, axis=0)
    mean_axis = mean_axis / np.linalg.norm(mean_axis)
    
    # Create the rotation matrix using the Rodrigues' rotation formula
    K = np.array([[0, -mean_axis[2], mean_axis[1]],
                  [mean_axis[2], 0, -mean_axis[0]],
                  [-mean_axis[1], mean_axis[0], 0]])
    R = np.eye(3) + np.sin(mean_angle) * K + (1 - np.cos(mean_angle)) * np.dot(K, K)
    
    return R


def best_fit_transform(A, B, window_size=10):
    # Apply moving average to smooth the trajectories
    smoothed_A = moving_average(A, window_size)
    smoothed_B = moving_average(B, window_size)
    
    # Calculate the directions of the trajectories
    directions_A = compute_directions(smoothed_A)
    directions_B = compute_directions(smoothed_B)

    # Compute the rotation matrix
    R = compute_rotation(directions_A, directions_B)

    # Rotate the head_trajectry_pred using the computed rotation matrix
    rotated_A = np.dot(A, R.T)

    # Compute the translation
    T = np.mean(B - rotated_A, axis=0)

    return R, T
import numpy as np
from .room import *
from ..pose2d import MediapipePose

KEYPOINT_DICT = MediapipePose.KEYPOINT_DICT


def extract_hmd_trajectory(trajectory_data, timestamps):
    """Extract synchronized HMD trajectory based on timestamps."""
    hmd_timestamps = np.array([transform['timestamp'] for transform in trajectory_data['transforms']])
    hmd_trajectory = np.array([[transform['position']['x'], transform['position']['y'], transform['position']['z']] for transform in trajectory_data['transforms']])
    
    # Use NumPy's searchsorted for efficient timestamp matching
    indices = np.searchsorted(hmd_timestamps, timestamps)
    hmd_trajectory_extracted = hmd_trajectory[indices]
    
    return hmd_trajectory_extracted


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


''' For synchronize timestamps '''

def find_time_offset(data1, data2, max_offset=10):    
    # Compute the cross-correlation and find the offset where it is maximized
    correlation = np.correlate(data1, data2, mode='full')
    offset = correlation.argmax() - (len(data1) - 1)
    
    # Clip the offset to the specified range
    offset = np.clip(offset, -max_offset, max_offset)
    return offset

def moving_average(data, window_size=5):
    cumsum = np.cumsum(data, axis=0)
    cumsum = np.vstack([np.zeros((1, 3)), cumsum])
    smoothed_data = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    return smoothed_data

def compute_velocity(data):
    deltas = np.diff(data, axis=0)
    velocities = np.linalg.norm(deltas, axis=1)
    return velocities

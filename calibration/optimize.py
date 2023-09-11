import cv2
import numpy as np
from scipy.optimize import least_squares   
from pose2d import MediapipePose

KEYPOINT_DICT = MediapipePose.KEYPOINT_DICT 


def optimize_func(x, keypoints2d_list, head_trajectry, K):
    n_frames, n_views, n_joints = keypoints2d_list.shape[:3]
    
    # カメラパラメータ計算
    new_camera_params = {}
    for i in range(n_views):
        t = np.array(x[i*6+3:i*6+6]).reshape([3,1])
        R = cv2.Rodrigues(np.array(x[i*6:i*6+3]))[0]
        Rc = R.T
        tc = -R.T @ t
        proj_matrix = K.dot(np.concatenate([Rc,tc], axis=-1))
        new_camera_params[i] = {
            'R': R,
            't': t,
            'proj_matrix': proj_matrix
        }
    
    # 3次元復元
    points3d = cv2.triangulatePoints(
        new_camera_params[0]['proj_matrix'],
        new_camera_params[1]['proj_matrix'],
        keypoints2d_list[0,:,:,:2].reshape([-1,2]).T,
        keypoints2d_list[1,:,:,:2].reshape([-1,2]).T
    )
    points3d = (points3d[:3,:] / points3d[3,:]).T
    keypoints3d = points3d.reshape([-1, 33, 3])
    
    head_trajectry_pred = keypoints3d[:, KEYPOINT_DICT['nose'], :]
    diff = head_trajectry_pred - head_trajectry
    squared_diff = diff ** 2
    mean_squared_diff = squared_diff.mean()
    rmse = np.sqrt(mean_squared_diff)
    return rmse

def optimize(camera_params, K):
    # 初期カメラパラメータ計算
    x0 = []
    for i in range(len(camera_params)):
        rotation = cv2.Rodrigues(camera_params[i]['R'])[0].flatten()
        translation = camera_params[i]['t'].flatten()
        x0.extend(rotation.tolist())
        x0.extend(translation.tolist())
    res = least_squares(optimize_func, x0, method='trf', loss='huber', verbose=1, diff_step=None, max_nfev=100)

    # カメラパラメータ計算
    new_camera_params = {}
    for i in range(len(camera_params)):
        t = np.array(res.x[i*6+3:i*6+6]).reshape([3,1])
        R = cv2.Rodrigues(np.array(res.x[i*6:i*6+3]))[0]
        Rc = R.T
        tc = -R.T @ t
        proj_matrix = K.dot(np.concatenate([Rc,tc], axis=-1))
        new_camera_params[i] = {
            'R': R,
            't': t,
            'proj_matrix': proj_matrix
        }
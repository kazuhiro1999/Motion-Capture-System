import cv2
import numpy as np


def estimate_initial_extrinsic(pts1, pts2, K):
    # pts : (N,2)
    E, mask = cv2.findEssentialMat(pts2, pts1, K, method=cv2.FM_LMEDS)
    pts, R, t, mask = cv2.recoverPose(E, pts2, pts1, K)
    return R, t


def calibrate_cameras(camera_settings, keypoints2d_list, base_i=0, pair_i=1, min_confidence=0.95):
    n_frames, n_views, n_joints, _ = keypoints2d_list.shape

    sample_points = []
    for frame_i in range(n_frames):
        for joint_i in range(n_joints):
            points = keypoints2d_list[frame_i,:,joint_i]
            if np.all(points[:,2] > min_confidence): # すべてのカメラで検出した点のみ使用
                sample_points.append(points[:,:2])
    sample_points = np.array(sample_points).transpose([1,0,2]) # shape:(n_views, n_points, 2)
    print(f"n_samples: {sample_points.shape[1]}")

    pts1 = sample_points[base_i].reshape([-1,2])
    pts2 = sample_points[pair_i].reshape([-1,2])
    K = camera_settings[base_i].intrinsic_matrix
    R, t = estimate_initial_extrinsic(pts1, pts2, K)

    camera_params = {
        base_i : {
            'R': np.eye(3,3),
            't': np.zeros([3,1]),
            'proj_matrix': get_projection_matrix(K, np.eye(3,3), np.zeros([3,1]))
        },
        pair_i : {
            'R': R,
            't': t,
            'proj_matrix': get_projection_matrix(K, R, t)
        }
    }

    # 3次元復元
    points3d = cv2.triangulatePoints(
        camera_params[base_i]['proj_matrix'],
        camera_params[pair_i]['proj_matrix'],
        keypoints2d_list[:,base_i,:,:2].reshape([-1,2]).T,
        keypoints2d_list[:,pair_i,:,:2].reshape([-1,2]).T
    )
    points3d = (points3d[:3,:] / points3d[3,:]).T

    keypoints3d = points3d.reshape([-1, 33, 3])  
    print('initial calibration end.')  
    return camera_params, keypoints3d


# 射影行列の計算
def get_projection_matrix(K, R, t):
    Rc = R.T
    tc = -R.T@t
    return K.dot(np.concatenate([Rc,tc], axis=-1))


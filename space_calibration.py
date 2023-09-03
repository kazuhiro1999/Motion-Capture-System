import json
import cv2
import numpy as np
from pose2d import MediapipePose
from scipy.spatial.transform import Rotation

KEYPOINT_DICT = MediapipePose.KEYPOINT_DICT


class SpaceCalibrator:

    def __init__(self, num_samples=300, min_confidence=0.5):
        self.camera_settings = []
        self.n_cameras = 0
        self.samples = []
        self.num_samples = num_samples
        self.min_confidence = min_confidence
        self.isActive = False

    def start_calibration(self, camera_settings):
        self.samples = []
        self.camera_settings = camera_settings
        self.n_cameras = len(camera_settings)
        self.isActive = True

    def add_samples(self, keypoints2d_list):
        keypoints2d_list = np.array(keypoints2d_list)
        n_views, n_joints, _ = keypoints2d_list.shape        
        if n_views != self.n_cameras:
            return
        self.samples.append(keypoints2d_list)

    def is_sampled(self):
        return len(self.samples) > self.num_samples
    
    def calibrate(self, base_i=0, pair_i=1):
        print("start calibration...")
        keypoints2d_list = np.array(self.samples)
        n_frames, n_views, n_joints, _ = keypoints2d_list.shape
        
        params, keypoints3d = calibrate_cameras(self.camera_settings, keypoints2d_list, base_i, pair_i) 
        self.isActive = False       
        print("calibration ended")
        return params, keypoints3d
    

def estimate_initial_extrinsic(pts1, pts2, K):
    # pts : (N,2)
    E, mask = cv2.findEssentialMat(pts2, pts1, K, method=cv2.FM_LMEDS)
    pts, R, t, mask = cv2.recoverPose(E, pts2, pts1, K)
    return R, t

def calibrate_cameras(camera_settings, keypoints2d_list, base_i=0, pair_i=1):
    n_frames, n_views, n_joints, _ = keypoints2d_list.shape

    sample_points = []
    for frame_i in range(n_frames):
        for joint_i in range(n_joints):
            points = keypoints2d_list[frame_i,:,joint_i]
            if np.all(points[:,2] > 0.95): # すべてのカメラで検出した点のみ使用
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

    # ルームキャリブレーション
    keypoints3d = points3d.reshape([-1, 33, 3])
    keypoints3d_list = keypoints3d[5:30]

    o_pos = determine_center_position(keypoints3d_list)
    o_mat = determine_forward_rotation(keypoints3d_list)
    o_rot = Rotation.from_matrix(o_mat)
    scale = determine_scale(keypoints3d_list, Height=1.6)

    for i, camera_setting in enumerate(camera_settings):
        t = o_rot.apply(camera_params[i]['t'].flatten() - o_pos).reshape([3,1]) * scale
        R = (o_rot * Rotation.from_matrix(camera_params[i]['R'])).as_matrix()
        Rc = R.T
        tc = -R.T@t
        camera_setting.extrinsic_matrix = np.concatenate([Rc,tc], axis=-1)
        camera_params[i] = {
            't': t.tolist(),
            'R': R.tolist(),
            'proj_matrix': get_projection_matrix(K, R, t).tolist()
        }

    # グローバル座標で3次元復元
    points3d = cv2.triangulatePoints(
        camera_settings[base_i].get_projection_matrix(),
        camera_settings[pair_i].get_projection_matrix(),
        keypoints2d_list[:,base_i,:,:2].reshape([-1,2]).T,
        keypoints2d_list[:,pair_i,:,:2].reshape([-1,2]).T
    )
    points3d = (points3d[:3,:] / points3d[3,:]).T
    keypoints3d = points3d.reshape([-1, 33, 3])
    return camera_params, keypoints3d


def determine_center_position(keypoints3d_list):
    # keypoints3d_list : (n_frames, n_joints, 3)
    l_foot = keypoints3d_list[:, KEYPOINT_DICT['left_ankle']]
    r_foot = keypoints3d_list[:, KEYPOINT_DICT['right_ankle']]
    m_foot = (l_foot + r_foot) / 2
    center_position = m_foot.mean(axis=0)
    return center_position

def determine_forward_rotation(keypoints3d_list):
    # keypoints3d_list : (n_frames, n_joints, 3)   
    l_shoulder = keypoints3d_list[:, KEYPOINT_DICT['left_shoulder']]
    r_shoulder = keypoints3d_list[:, KEYPOINT_DICT['right_shoulder']]
    l_hips = keypoints3d_list[:, KEYPOINT_DICT['left_hip']]
    r_hips = keypoints3d_list[:, KEYPOINT_DICT['right_hip']]    
    m_hips = (l_hips + r_hips) / 2
    forward_vector = np.cross(l_shoulder - m_hips, r_shoulder - m_hips).mean(axis=0)
    forward = forward_vector / np.linalg.norm(forward_vector)
    rotation_matrix = (forward.reshape([3,1]) @ np.array([0,0,1]).reshape([1,3])).astype(np.float32)
    '''
    neck = (l_shoulder + r_shoulder) / 2
    root = (l_ankle + r_ankle) / 2
    l_ankle = keypoints3d_list[:, KEYPOINT_DICT['left_ankle']]
    r_ankle = keypoints3d_list[:, KEYPOINT_DICT['right_ankle']]
    up_vector = (neck - root).mean(axis=0)
    up = up_vector / np.linalg.norm(up_vector)
    #rotation_matrix = (up_vector.reshape([3,1]) @ np.array([0,1,0]).reshape([1,3])).astype(np.float32)

    # 右ベクトルを計算
    right_vector = np.cross(up, forward)
    right = right_vector / np.linalg.norm(right_vector)
    
    # 正確な上方向ベクトルを再計算
    up = np.cross(forward, right)
    
    # 回転行列を作成
    rotation_matrix = np.array([
        [right[0], right[1], right[2]],
        [up[0], up[1], up[2]],
        [-forward[0], -forward[1], -forward[2]]
    ])
    '''    
    return rotation_matrix

def determine_scale(keypoints3d_list, Height=1.0):
    head = keypoints3d_list[:, KEYPOINT_DICT['nose']]
    l_foot = keypoints3d_list[:, KEYPOINT_DICT['left_ankle']]
    r_foot = keypoints3d_list[:, KEYPOINT_DICT['right_ankle']]
    m_foot = (l_foot + r_foot) / 2
    height = np.linalg.norm(head - m_foot, axis=-1).mean()
    scale = Height / height
    return scale

def get_projection_matrix(K, R, t):
    Rc = R.T
    tc = -R.T@t
    return K.dot(np.concatenate([Rc,tc], axis=-1))


if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor
    from camera import USBCamera
    from utils import TimeUtil
    from visalization import draw_keypoints
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    with open('./config_2.json', 'r') as f:
        config = json.load(f)

    # Initialize Motion Capture
    cameras = [USBCamera(camera_config) for camera_config in config['cameras']]
    for camera in cameras:
        camera.open()

    pose_estimators = [MediapipePose() for _ in config['cameras']]

    keypoints2d_list = []
    
    calibrator = SpaceCalibrator()
    calibrator.start_calibration([camera.camera_setting for camera in cameras])

    t = TimeUtil.get_unixtime()
    print("capture start")
    # Main loop
    with ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            timestamp = t
            # send 2d pose estimation
            t = TimeUtil.get_unixtime()
            frames = []
            futures = []
            for camera, pose_estimator in zip(cameras, pose_estimators):
                frame = camera.get_image()
                if frame is not None:
                    frames.append(frame)
                    future = executor.submit(pose_estimator.process, frame)
                    futures.append(future)

            # get 2d pose estimation results
            keypoints2d_list = []

            for camera, frame, pose_estimator, future in zip(cameras, frames, pose_estimators, futures):
                keypoints2d = future.result()
                debug_image = draw_keypoints(frame, keypoints2d, MediapipePose.KINEMATIC_TREE)
                cv2.imshow(camera.name, debug_image)
                if keypoints2d is not None:
                    keypoints2d_list.append(keypoints2d)

            if calibrator.isActive:
                calibrator.add_samples(keypoints2d_list)
                print(len(calibrator.samples))
                if calibrator.is_sampled():
                    params, keypoints3d_list = calibrator.calibrate()


            if cv2.waitKey(1) == 27:  # Press ESC to exit
                break

    # End Motion Caapture
    for camera in cameras:
        camera.close()

    cv2.destroyAllWindows()
    
    # save calibration result
    result = {
        'camera_settings': params,
        'keypoints3d': keypoints3d_list.tolist()
    }
    with open('./data/result.json', 'w') as f:
        json.dump(result, f, indent=4)

    # visualize calibration result    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for keypoints3d in keypoints3d_list:
        ax.cla()
        ax.set_xlim(1.5,-1.5)
        ax.set_ylim(-1.5,1.5)
        ax.set_zlim(0,3)
        for chain in MediapipePose.KINEMATIC_TREE:
            for j in range(len(chain)-1):
                x1,y1,z1 = keypoints3d[chain[j]]
                x2,y2,z2 = keypoints3d[chain[j+1]]
                ax.plot([x1,x2],[z1,z2],[y1,y2], color='black')     
        ax.scatter([0] [0], [0], s=30, color='black', marker='x')
        for camera in cameras:
            Rc = camera.camera_setting.extrinsic_matrix[:,:3]
            tc = camera.camera_setting.extrinsic_matrix[:,3:]
            R = Rc.T
            t = -Rc.T@tc
            ax.scatter(t[0,0], t[2,0], t[1,0], s=30, color='blue')
            tt = R @ np.array([[0],[0],[0.5]]) + t
            ax.plot([t[0,0], tt[0,0]], [t[2,0], tt[2,0]], [t[1,0], tt[1,0]], color='blue')
        plt.draw()
        plt.pause(0.01) 
        plt.cla()
import json
import cv2
import numpy as np
from calibration.extrinsic import calibrate_cameras, get_projection_matrix
from calibration.room import *
from pose2d import MediapipePose
from visalization import draw_camera

KEYPOINT_DICT = MediapipePose.KEYPOINT_DICT


class SpaceCalibrator:

    def __init__(self, num_samples=300, min_confidence=0.5, reference_height=1.6):
        self.camera_settings = []
        self.n_cameras = 0
        self.samples = []
        self.num_samples = num_samples
        self.min_confidence = min_confidence
        self.reference_height = reference_height
        self.isActive = False

    def start_calibration(self, camera_settings):
        self.samples = []
        self.camera_settings = camera_settings
        self.n_cameras = len(camera_settings)
        self.isActive = True

    def add_samples(self, keypoints2d_list):
        if len(keypoints2d_list) != self.n_cameras:
            return
        keypoints2d_list = np.array(keypoints2d_list)
        self.samples.append(keypoints2d_list)

    def is_sampled(self):
        return len(self.samples) > self.num_samples
    
    def calibrate(self, base_i=0, pair_i=1):
        print("start calibration...")
        keypoints2d_list = np.array(self.samples)
        n_frames, n_views, n_joints, _ = keypoints2d_list.shape
        
        camera_params, keypoints3d = calibrate_cameras(self.camera_settings, keypoints2d_list, base_i, pair_i) 
        keypoints3d_list = keypoints3d[5:30]

        # ルームキャリブレーション用の初期パラメータ
        s0 = determine_scale(keypoints3d_list, Height=self.reference_height)
        p0 = determine_center_position(keypoints3d_list)
        forward = determine_forward_vector(keypoints3d_list)
        up = determine_upward_vector(keypoints3d_list)
        R0 = rotation_matrix_from_vectors(forward, up)
        t0 = -R0 @ (p0 * s0) 

        for i, camera_setting in enumerate(self.camera_settings):
            R = R0 @ camera_params[i]['R']
            t = R0 @ (s0 * camera_params[i]['t']) + t0.reshape([3,1])         
            Rc = R.T
            tc = -R.T@t
            camera_setting.extrinsic_matrix = np.concatenate([Rc,tc], axis=-1)
            camera_params[i] = {
                't': t.tolist(),
                'R': R.tolist(),
                'proj_matrix': get_projection_matrix(camera_setting.intrinsic_matrix, R, t).tolist()
            }

        # グローバル座標で3次元復元
        points3d = cv2.triangulatePoints(
            self.camera_settings[base_i].get_projection_matrix(),
            self.camera_settings[pair_i].get_projection_matrix(),
            keypoints2d_list[:,base_i,:,:2].reshape([-1,2]).T,
            keypoints2d_list[:,pair_i,:,:2].reshape([-1,2]).T
        )
        points3d = (points3d[:3,:] / points3d[3,:]).T
        keypoints3d = points3d.reshape([-1, 33, 3])
        
        self.isActive = False       
        return camera_params, keypoints3d


def calibration_process(config_path, height=1.6, output_dir=None):
    from concurrent.futures import ThreadPoolExecutor
    from camera import USBCamera
    from visalization import draw_keypoints
    from utils import TimeUtil

    t_list = []
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Initialize Motion Capture
    cameras = [USBCamera(camera_config) for camera_config in config['cameras']]
    for camera in cameras:
        camera.open()

    # if save videos
    if output_dir is not None:
        writers = []
        for camera in cameras:
            path = f'{output_dir}/{camera.name}.mp4'
            fmt = cv2.VideoWriter_fourcc('m','p','4','v')
            writer = cv2.VideoWriter(path, fmt, 30.0, (960,540))
            writers.append(writer)


    pose_estimators = [MediapipePose(model_complexity=2) for _ in config['cameras']]

    keypoints2d_list = []
    
    calibrator = SpaceCalibrator(reference_height=height)
    calibrator.start_calibration([camera.camera_setting for camera in cameras])

    print("sampling...")
    # Main loop
    with ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            t = TimeUtil.get_unixtime()
            t_list.append(t)
            # send 2d pose estimation
            frames = []
            futures = []
            for i, (camera, pose_estimator) in enumerate(zip(cameras, pose_estimators)):
                frame = camera.get_image()
                if frame is not None:
                    frames.append(frame)
                    future = executor.submit(pose_estimator.process, frame)
                    futures.append(future)
                    # save video
                    if output_dir is not None:
                        writers[i].write(frame)

            # get 2d pose estimation results
            keypoints2d_list = []

            for camera, frame, pose_estimator, future in zip(cameras, frames, pose_estimators, futures):
                keypoints2d = future.result()
                debug_image = draw_keypoints(frame, keypoints2d, MediapipePose.KINEMATIC_TREE)
                cv2.imshow(camera.name, debug_image)
                if keypoints2d is not None:
                    keypoints2d_list.append(keypoints2d)

            # add sample
            if calibrator.isActive:
                calibrator.add_samples(keypoints2d_list)
                if calibrator.is_sampled():
                    params, keypoints3d_list = calibrator.calibrate()
                    print("calibration ended")
                    break

            if cv2.waitKey(1) == 27:  # Press ESC to exit
                print("calibration stopped")
                break

    # End 
    for camera in cameras:
        camera.close()

    cv2.destroyAllWindows()
    
    # save calibration result
    for camera in cameras:
        camera.camera_setting.save()

    if output_dir is not None:
        result = {
            'camera_settings': params,
            'timestamps': t_list,
            'keypoints3d': keypoints3d_list.tolist()
        }
        with open(f'{output_dir}/result.json', 'w') as f:
            json.dump(result, f, indent=4)

        for writer in writers:
            writer.release()

    return [camera.camera_setting for camera in cameras], keypoints3d_list 


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    import PySimpleGUI as sg
    output_dir = sg.popup_get_folder("保存先を選んでね")

    # calibration
    camera_settings, keypoints3d_list = calibration_process(config_path="config.json", output_dir=output_dir)

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
        for camera_setting in camera_settings:
            Rc = camera_setting.extrinsic_matrix[:,:3]
            tc = camera_setting.extrinsic_matrix[:,3:]
            R = Rc.T
            t = -Rc.T@tc
            draw_camera(ax, R, t, color='blue')
            
        plt.draw()
        plt.pause(0.01) 
        plt.cla()
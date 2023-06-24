import cv2
import numpy as np
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from camera import USBCamera
from pose3d import recover_pose_3d
from pose2d import MediapipePose
from network import UDPClient
from utils import TimeUtil


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config.json")

    args = parser.parse_args()
    return args



def main():
    args = get_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    cameras = [USBCamera(camera_config) for camera_config in config['cameras']]
    for camera in cameras:
        camera.open()

    pose_estimators = [MediapipePose() for _ in config['cameras']]

    udp_client = UDPClient(host='127.0.0.1', port=50000)
    udp_client.open()

    keypoints2d_list = []
    proj_matrices = []
    t = TimeUtil.get_unixtime()

    with ThreadPoolExecutor(max_workers=4) as executor:

        while True:
            timestamp = t

            # send 2d pose estimation
            t = TimeUtil.get_unixtime()
            futures = []
            for camera, pose_estimator in zip(cameras, pose_estimators):
                frame = camera.get_image()
                if frame is not None:
                    cv2.imshow(camera.name, cv2.resize(frame, dsize=(640,360)))
                    future = executor.submit(pose_estimator.process, frame)
                    futures.append(future)

            # 3d pose estimation 
            future_keypoints3d = executor.submit(recover_pose_3d, proj_matrices, keypoints2d_list)

            keypoints3d = future_keypoints3d.result()
            if keypoints3d is not None:
                data = {"Type":MediapipePose.Type, "TimeStamp": timestamp, "Bones":[]}
                keys = MediapipePose.KEYPOINT_DICT
                for key in keys:
                    bone = {
                        "Name": key,
                        "Position":{
                            "x": float(keypoints3d[keys[key],0]),
                            "y": float(keypoints3d[keys[key],1]),
                            "z": float(keypoints3d[keys[key],2]),
                        }
                    }
                    data['Bones'].append(bone)                
                ret = udp_client.send(data)


            # get 2d pose estimation results
            keypoints2d_list = []
            proj_matrices = []

            for camera, pose_estimator, future in zip(cameras, pose_estimators, futures):
                keypoints2d = future.result()
                if keypoints2d is not None:
                    proj_matrix = camera.camera_setting.get_projection_matrix()
                    proj_matrices.append(proj_matrix)
                    keypoints2d_list.append(keypoints2d)


            if cv2.waitKey(1) == 27:  # Press ESC to exit
                break


    for camera in cameras:
        camera.close()

    udp_client.close()

    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    main()

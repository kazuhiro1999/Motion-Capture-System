'''
キャリブレーションの結果を評価
'''
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from camera import CameraSetting
from pose2d import MediapipePose


def tangent_angle(u: np.ndarray, v: np.ndarray):
    i = np.inner(u, v)
    n = np.linalg.norm(u) * np.linalg.norm(v)
    c = i / n
    return np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))


with open('./config1.json', 'r') as f:
    config = json.load(f)

with open('./data/result.json', 'r') as f:
    result = json.load(f)

camera_settings = [CameraSetting(camera_config['setting_path']) for camera_config in config['cameras']]


for i, camera_setting in enumerate(camera_settings):
    # カメラ（GT）
    Rc = camera_setting.extrinsic_matrix[:,:3]
    tc = camera_setting.extrinsic_matrix[:,3:]
    R = Rc.T
    t = -Rc.T@tc
    tt = R @ np.array([[0],[0],[1]]) + t   
    # カメラ（推定）
    R1 = np.array(result['camera_settings'][str(i)]['R'])
    t1 = np.array(result['camera_settings'][str(i)]['t'])
    tt1 = R1 @ np.array([[0],[0],[1]]) + t1
    # 誤差
    diff = np.linalg.norm(t1.flatten() - t.flatten())
    angle = tangent_angle((tt1 - t1).flatten(), (tt - t).flatten())
    print(f"camera{i}: diff={diff}, angle={angle}")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

exit_loop = False
while not exit_loop:
    for keypoints3d in result['keypoints3d']:
        ax.cla()
        ax.set_xlim(1.5,-1.5)
        ax.set_ylim(-1.5,1.5)
        ax.set_zlim(0,3)
        # モーション
        for chain in MediapipePose.KINEMATIC_TREE:
            for j in range(len(chain)-1):
                x1,y1,z1 = keypoints3d[chain[j]]
                x2,y2,z2 = keypoints3d[chain[j+1]]
                ax.plot([x1,x2],[z1,z2],[y1,y2], color='black')     
        # 原点
        ax.scatter([0] [0], [0], s=30, color='black', marker='x')
        # カメラ（GT）
        for camera_setting in camera_settings:
            Rc = camera_setting.extrinsic_matrix[:,:3]
            tc = camera_setting.extrinsic_matrix[:,3:]
            R = Rc.T
            t = -Rc.T@tc
            ax.scatter(t[0,0], t[2,0], t[1,0], s=30, color='green')
            tt = R @ np.array([[0],[0],[0.5]]) + t
            ax.plot([t[0,0], tt[0,0]], [t[2,0], tt[2,0]], [t[1,0], tt[1,0]], color='green')        
        # カメラ（推定）
        for i in result['camera_settings']:
            R = np.array(result['camera_settings'][i]['R'])
            t = np.array(result['camera_settings'][i]['t'])
            ax.scatter(t[0,0], t[2,0], t[1,0], s=30, color='blue')
            tt = R @ np.array([[0],[0],[0.5]]) + t
            ax.plot([t[0,0], tt[0,0]], [t[2,0], tt[2,0]], [t[1,0], tt[1,0]], color='blue')

        plt.draw()
        plt.pause(0.01) 
        if not plt.fignum_exists(1):
            exit_loop = True
            break
        plt.cla()

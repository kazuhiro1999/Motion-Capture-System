'''
キャリブレーションの結果を評価
'''
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from camera import CameraSetting
from pose2d import MediapipePose
import PySimpleGUI as sg

from visalization import draw_camera


def riemannian_distance(R, Rg):
    """
    Compute the Riemannian distance between two rotation matrices R and Rg.
    """
    # Compute the matrix logarithm of the product of R transpose and Rg
    phi = np.arccos((np.trace(np.dot(R.T, Rg)) - 1) / 2)
    log_value = (phi / (2 * np.sin(phi))) * (R - R.T)
    
    # Compute the Frobenius norm of the log_value
    frobenius_norm = np.linalg.norm(log_value, 'fro')
    
    # Return the Riemannian distance
    return (1/np.sqrt(2)) * frobenius_norm


with open('./config1.json', 'r') as f:
    config = json.load(f)

filepath = sg.popup_get_file("評価するファイルを選んでね")
with open(filepath, 'r') as f:
    result = json.load(f)

camera_settings = [CameraSetting(camera_config['setting_path']) for camera_config in config['cameras']]


for i, camera_setting in enumerate(camera_settings):
    # カメラ（GT）
    Rc = camera_setting.extrinsic_matrix[:,:3]
    tc = camera_setting.extrinsic_matrix[:,3:]
    Rg = Rc.T
    tg = -Rc.T@tc
    # カメラ（推定）
    R = np.array(result['camera_settings'][str(i)]['R'])
    t = np.array(result['camera_settings'][str(i)]['t'])
    # 誤差
    Et = np.linalg.norm(t.flatten() - tg.flatten())
    Er = riemannian_distance(R, Rg)
    print(f"camera{i}: E(t)={Et:.3f}, E(R)={Er:.3f}")

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
            draw_camera(ax, R, t, 'green')       
        # カメラ（推定）
        for i in result['camera_settings']:
            R = np.array(result['camera_settings'][i]['R'])
            t = np.array(result['camera_settings'][i]['t'])
            draw_camera(ax, R, t, 'blue')   

        plt.draw()
        if not plt.fignum_exists(1):
            exit_loop = True
            break        
        plt.pause(0.01) 
        plt.cla()

import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from motioncapture.capture import MotionCapture
from motioncapture.pose2d import MediapipePose
from motioncapture.visalization import draw_keypoints3d


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config.json")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, args.config)

    capture = MotionCapture(config_path)
    capture.start()
    
    # visualize calibration result    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    while True:
        timestamp, keypoints2d_list, keypoints3d = capture.read()
        
        ax.cla()
        ax.set_xlim(-1.5,1.5)
        ax.set_ylim(-1.5,1.5)
        ax.set_zlim(0,3)
        
        draw_keypoints3d(ax, keypoints3d, MediapipePose.KINEMATIC_TREE)
            
        plt.draw()
        plt.pause(0.01) 
        if not plt.fignum_exists(1):
            plt.clf()
            plt.close()
            break        

    capture.end()


if __name__ == '__main__':
    main()

import cv2


def draw_keypoints(image, keypoints2d, skeleton=None, th=0.2):
    if keypoints2d is None:
        return image
    debug_image = image.copy()
    for x, y, confidence in keypoints2d:
        if confidence >= th:
            cv2.circle(debug_image, (int(x), int(y)), radius=5, color=(0,255,0), thickness=5)
    if skeleton is None:
        return debug_image

    for chain in skeleton:
        for j in range(len(chain)-1):
            x1,y1,c1 = keypoints2d[chain[j]]
            x2,y2,c2 = keypoints2d[chain[j+1]]
            if c1 >= th and c2 >= th:
                cv2.line(debug_image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0,255,0), thickness=2)

    return debug_image


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def draw_keypoints3d(keypoints3d, skeleton=None):
    if keypoints3d is None or skeleton is None:
        return
    ax.cla()
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(0,2)
    for chain in skeleton:
        for j in range(len(chain)-1):
            x1,y1,z1 = keypoints3d[chain[j]]
            x2,y2,z2 = keypoints3d[chain[j+1]]
            ax.plot([x1,x2],[z1,z2],[y1,y2], color='black')
    plt.pause(0.001) 
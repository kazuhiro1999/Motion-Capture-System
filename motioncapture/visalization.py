import cv2
import numpy as np


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


def draw_keypoints3d(ax, keypoints3d, skeleton=None):    
    if keypoints3d is None or skeleton is None:
        return ax
    for chain in skeleton:
        for j in range(len(chain)-1):
            x1,y1,z1 = keypoints3d[chain[j]]
            x2,y2,z2 = keypoints3d[chain[j+1]]
            ax.plot([x1,x2],[z1,z2],[y1,y2], color='black')
    return ax


def draw_camera(ax, R, t, color='blue'):
    o = t.flatten()
    p1 = (R @ np.array([[0.2],[0.2],[0.5]]) + t).flatten()
    p2 = (R @ np.array([[0.2],[-0.2],[0.5]]) + t).flatten()
    p3 = (R @ np.array([[-0.2],[0.2],[0.5]]) + t).flatten()
    p4 = (R @ np.array([[-0.2],[-0.2],[0.5]]) + t).flatten()
    ax.scatter(o[0], o[2], o[1], s=30, color=color)
    ax.plot([o[0], p1[0]], [o[2], p1[2]], [o[1], p1[1]], color=color)
    ax.plot([o[0], p2[0]], [o[2], p2[2]], [o[1], p2[1]], color=color)
    ax.plot([o[0], p3[0]], [o[2], p3[2]], [o[1], p3[1]], color=color)
    ax.plot([o[0], p4[0]], [o[2], p4[2]], [o[1], p4[1]], color=color)
    ax.plot([o[0], p4[0]], [o[2], p4[2]], [o[1], p4[1]], color=color)
    ax.plot([p1[0], p2[0], p4[0], p3[0], p1[0]], [p1[2], p2[2], p4[2], p3[2], p1[2]], [p1[1], p2[1], p4[1], p3[1], p1[1]], color=color)
    return ax
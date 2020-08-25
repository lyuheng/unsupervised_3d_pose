
import numpy as np
import cv2
import matplotlib.pyplot as plt

import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D

def show_2d(img, points, color, edges):
    num_joints = points.shape[0]

    points = (points.reshape(num_joints, -1)).astype(np.int32)

    for j in range(num_joints):
        cv2.circle(img, (points[j,0], points[j,1]), radius=3, color=color, thickness=-1)
    
    for e in edges:
        if points[e].min() > 0:
            cv2.line(img, (points[e[0], 0], points[e[0], 1]),
                          (points[e[1], 0], points[e[1], 1]), color=color, thickness=-1)
    return img
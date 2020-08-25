import os
import os.path as osp
import numpy as np
import json
import cv2
import shutil
import pickle as pkl

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl



# For Human3.6 Dataset, 17 joints
skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), 
            (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )


def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines=skeleton, filename=None, path='./eval', epoch=0):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1,0], kpt_3d[i2,0]])
        y = np.array([kpt_3d[i1,1], kpt_3d[i2,1]])
        z = np.array([kpt_3d[i1,2], kpt_3d[i2,2]])

        if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1,0] > 0:
            ax.scatter(kpt_3d[i1,0], kpt_3d[i1,2], -kpt_3d[i1,1], c=colors[l], marker='o')
        if kpt_3d_vis[i2,0] > 0:
            ax.scatter(kpt_3d[i2,0], kpt_3d[i2,2], -kpt_3d[i2,1], c=colors[l], marker='o')

    x_r = np.array([0, 256], dtype=np.float32)
    y_r = np.array([0, 256], dtype=np.float32)
    z_r = np.array([0, 1], dtype=np.float32)
    
    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    ax.set_zlim([-2,2])
    ax.legend()

    #plt.show()
    plt.savefig(path+'/eval_'+str(epoch))
    #cv2.waitKey(0)
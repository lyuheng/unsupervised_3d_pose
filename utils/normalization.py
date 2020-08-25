
import torch
import numpy as np

import os

def normalize_3d(pose):
    """
    pose: (num_joints, 3)  ndarray
    """

    xs = pose[:,0] - pose[0,0]
    ys = pose[:,1] - pose[0,1]
    ls= np.sqrt(xs[1:]**2 + ys[:1]**2)
    scale = ls.mean(axis=0)
    pose = pose/scale
    pose[:,0] -= pose



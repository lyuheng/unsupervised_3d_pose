
import argparse
import time
import numpy as np
import pickle as pkl
import os
import os.path as osp
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

from utils.logger import Logger
from utils.misc import AverageMeter, mkdir_p, save_checkpoint, str_to_bool
from utils.vis import vis_3d_skeleton

from models.model import Lifter, Discriminator
from datasets.H36M import H36M


parser = argparse.ArgumentParser(description='PyTorch Evaluation Human3.6M')

parser.add_argument('--model_path', default='/home/lyuheng/vision/unsupervised_3d_pose_lift/checkpoints/checkpoint_50000.pth.tar',
                    help='model path')

parser.add_argument('--demo_dir', default='/home/lyuheng/vision/unsupervised_3d_pose_lift/demo',
                    help='demo path')
args = parser.parse_args()

if not osp.isdir(args.demo_dir):
    mkdir_p(args.demo_dir)



with open('./data/sh_detect_2d.pkl', 'rb') as f:
    p2d_sh = pkl.load(f)


L = Lifter()

def load_model(model, path):
    state = torch.load(path)
    model.load_state_dict(state['state_dict_L'])

def normalize_2d(pose):
    """
    pose: (length=1, 17*2)
    Return:
        pose.T (length=1, 17*2)
    """
    xs = pose.T[0::2] - pose.T[0]
    ys = pose.T[1::2] - pose.T[1]
    pose = pose.T / np.sqrt(xs[1:] ** 2 + ys[1:] ** 2).mean(axis=0)
    mu_x = pose[0].copy()
    mu_y = pose[1].copy()
    pose[0::2] -= mu_x
    pose[1::2] -= mu_y
    return pose.T

load_model(L, args.model_path)

L.eval()

#print(p2d_sh['S9']['Directions'].keys())
xys = p2d_sh['S9']['Directions']['54138969'] # subact=2 cam=2
"""
for i in range(xys.shape[0]):
    xy = xys[i:i+1,:]
    xy = normalize_2d(xy)
    xy = torch.from_numpy(xy)
    z_pred = L(xy)
    x = xy[:, 0::2]
    y = xy[:, 1::2]
    joint_3d = torch.cat((x[:,:,None], y[:,:,None], z_pred[:,:,None]), dim=2)
    joint_3d = joint_3d.view(17,3).cpu().detach().numpy()
    vis = np.ones((17,1))
    vis_3d_skeleton(joint_3d, vis, path=args.demo_dir, epoch='{:04}'.format(i))
"""

# make GIF file
import imageio

'''
outfilename = 'final.gif' 
filenames = []
for i in range(999,1999,10):
    filename = osp.join(args.demo_dir, 'eval_{:04}.png'.format(i))
    filenames.append(filename)
frames = []
for image_name in filenames:
    im = cv2.imread(image_name)[:,:,::-1]
    frames.append(im)
imageio.mimsave(outfilename, frames, 'GIF', duration=0.25)
'''

outfilename = 'video.gif' 
filenames = []
for i in range(1000, 2000, 10):
    filename = osp.join('/media/lyuheng/Seagate Portable Drive/Human36m/images/s_09_act_02_subact_02_ca_01', 's_09_act_02_subact_02_ca_01_{:06}.jpg'.format(i))
    filenames.append(filename)
frames = []
for image_name in filenames:
    im = cv2.imread(image_name)[:,:,::-1]
    im = cv2.resize(im, (256,256))
    frames.append(im)
imageio.mimsave(outfilename, frames, 'GIF', duration=0.25)


import copy
import os
import os.path as osp
import pickle as pkl
import numpy as np

import torch
import torch.utils.data as data

# 1211: 1383


# Joints in H3.6M -- data has 32 joints,
# but only 17 that move; these are the indices.

H36M_NAMES = [''] * 32
H36M_NAMES[0] = 'Hip'
H36M_NAMES[1] = 'RHip'
H36M_NAMES[2] = 'RKnee'
H36M_NAMES[3] = 'RFoot'
H36M_NAMES[6] = 'LHip'
H36M_NAMES[7] = 'LKnee'
H36M_NAMES[8] = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'


"""
dict_keys(['Directions', 'WalkingDog', 'Eating 2', 'Posing', 'Smoking 1', 'Phoning', 
'TakingPhoto 1', 'Waiting', 'Walking 1', 'Purchases 1', 'Eating', 'Phoning 1', 'TakingPhoto', 
'Directions 1', 'SittingDown 2', 'Discussion 1', 'Posing 1', 'WalkTogether 1', 'Purchases', 
'SittingDown', 'Smoking', 'Greeting 1', 'Sitting 1', 'Sitting 2', 'WalkTogether', 'Waiting 1', 
'WalkingDog 1', 'Greeting', 'Discussion', 'Walking'])
"""


def project_point_radial(P, R, T, f, c, k, p):
    """
    Project points from 3d to 2d using camera parameters
    including radial and tangential distortion
    Args
    P: Nx3 points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
    Returns
    Proj: Nx2 points in pixel space
    D: 1xN depth of each point in camera space
    radial: 1xN radial distortion per point
    tan: 1xN tangential distortion per point
    r2: 1xN squared radius of the projected points before distortion
    """

    # P is a matrix of 3-dimensional points
    assert len(P.shape) == 2
    assert P.shape[1] == 3

    N = P.shape[0]
    X = R.dot(P.T - T)  # rotate and translate
    XX = X[:2, :] / X[2, :]
    r2 = XX[0, :] ** 2 + XX[1, :] ** 2

    radial = 1 + np.einsum(
        'ij,ij->j', np.tile(k, (1, N)), np.array([r2, r2 ** 2, r2 ** 3]))
    tan = p[0] * XX[1, :] + p[1] * XX[0, :]

    XXX = XX * np.tile(radial + tan, (2, 1)) + \
        np.outer(np.array([p[1], p[0]]).reshape(-1), r2)

    Proj = (f * XXX) + c
    Proj = Proj.T

    D = X[2,]

    return Proj, D, radial, tan, r2


class H36M(data.Dataset):

    """
    Dataset for Human3.6M
    """

    def __init__(self, length, 
                        action='all', 
                        is_train=True, 
                        use_sh_detection=True,
                        ):

        if is_train:
            subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
        else:
            subjects = ['S9', 'S11']

        if not osp.exists('./data'):
            os.mkdir('./data')
        
        if not os.path.exists('./data/points_3d.pkl'):
            print('Downloading 3D points in Human3.6M dataset.')
            os.system('wget --no-check-certificate "https://onedriv' + \
                'e.live.com/download?cid=B08D60FE71FF90FD&resid=B08' + \
                'D60FE71FF90FD%2118616&authkey=AFIfEB6VYEZnhlE" -O ' + \
                'data/points_3d.pkl')

        with open('./data/points_3d.pkl', 'rb') as f:
            p3d = pkl.load(f)
        
        if not os.path.exists('./data/cameras.pkl'):
            print('Downloading camera parameters.')
            os.system('wget --no-check-certificate "https://onedriv' + \
                'e.live.com/download?cid=B08D60FE71FF90FD&resid=B08' + \
                'D60FE71FF90FD%2118615&authkey=AEUoi3s16rBTFRA" -O ' + \
                'data/cameras.pkl')
    
        with open('./data/cameras.pkl', 'rb') as f:
            cams = pkl.load(f)

        if use_sh_detection:
            if not os.path.exists('./data/sh_detect_2d.pkl'):
                print('Downloading detected 2D points by Stacked Hourglass.')
                os.system('wget --no-check-certificate "https://onedriv' + \
                    'e.live.com/download?cid=B08D60FE71FF90FD&resid=B08' + \
                    'D60FE71FF90FD%2118619&authkey=AMBf6RPcWQgjsh0" -O ' + \
                    'data/sh_detect_2d.pkl')
            with open('./data/sh_detect_2d.pkl', 'rb') as f:
                p2d_sh = pkl.load(f)
            
        
        with open('./data/actions.txt') as f:
            actions_all = f.read().split('\n')[:-1]
        if action == 'all':
            actions = actions_all
        elif action in actions_all:
            actions = [action]
        else:
            raise Exception('Invalid action.')

        dim_to_use_x = np.where(np.array([x != '' for x in H36M_NAMES]))[0] * 3
        dim_to_use_y = dim_to_use_x + 1
        dim_to_use_z = dim_to_use_x + 2
        dim_to_use = np.array(
            [dim_to_use_x, dim_to_use_y, dim_to_use_z]).T.flatten() 
        # [ 0  1  2  3  4  5  6  7  8  9 10 11 18 19 20 21 22 23 24 25 26 36 37 38 39 40 41 42 43 44 45 46 47 51 52 53 54 55 56 57 58 59 75 76 77 78 79 80 81 82 83]
        self.N = len(dim_to_use_x)  # 17

        p3d = copy.deepcopy(p3d)
        self.data_list = []

        for s in subjects:
            for action_name in actions:

                def _search(a):
                    fs = list(filter(
                        lambda x: x.split()[0] == a, p3d[s].keys()
                    ))
                    return fs
                files = []
                files += _search(action_name)

                # 'Photo' is 'TakingPhoto' in S1
                if action_name == 'Photo':
                    files += _search('TakingPhoto')
                # 'WalkDog' is 'WalkingDog' in S1
                if action_name == 'WalkDog':
                    files += _search('WalkingDog')
                
                for file_name in files:
                    p3d[s][file_name] = p3d[s][file_name][:, dim_to_use]  # (N, 17*3)
                    L = p3d[s][file_name].shape[0]  # N=L
                    for cam_name in cams[s].keys():
                        if not (cam_name == '54138969' and s == 'S11' \
                                and action_name == 'Directions'):
                            # 50fps -> 25fps
                            for start_pos in range(0, L-length+1, 2):
                                info = {
                                    'subject': s,
                                    'action_name': action_name,
                                    'start_pos': start_pos,
                                    'cam_name': cam_name,
                                    'length': length,
                                    'file_name': file_name,
                                    'dataset': 'Human3.6M',
                                }
                                self.data_list.append(info)
        self.p3d = p3d
        self.cams = cams
        self.is_train = is_train        
        self.use_sh_detection = use_sh_detection
        if use_sh_detection:
            self.p2d_sh = p2d_sh
        self.length = length

                

    def __getitem__(self, index):

        info = self.data_list[index]
        subject = info['subject']
        start_pos = info['start_pos']
        length = info['length']
        cam_name = info['cam_name']
        file_name = info['file_name']

        pos_xyz = self.p3d[subject][file_name][start_pos:start_pos+length] # (length,17*3)
        params = self.cams[subject][cam_name]

        if self.use_sh_detection:
            if 'TakingPhoto' in file_name:
                file_name = file_name.replace('TakingPhoto', 'Photo')
            if 'WalkingDog' in file_name:
                file_name = file_name.replace('WalkingDog', 'WalkDog')
            sh_detect_xy = self.p2d_sh[subject][file_name]
            sh_detect_xy = sh_detect_xy[cam_name][start_pos:start_pos+length]
        
        P = pos_xyz.reshape(-1,3)  # (17*length, 3)
        X = params['R'].dot(P.T).T  # (17*length, 3) # world2cam
        X = X.reshape(-1,self.N * 3) # (length,17*3)

        X, scale = normalize_3d(X)
        X = X.astype(np.float32)
        scale = scale.astype(np.float32)

        if self.use_sh_detection:
            sh_detect_xy = normalize_2d(sh_detect_xy)
            sh_detect_xy = sh_detect_xy.astype(np.float32)
            return sh_detect_xy, X, scale
        else:
            # TODO:
            proj = project_point_radial(P, **params)[0]
            proj = proj.reshape(-1, self.N * 2)  # shape=(length, 2*n_joints)
            proj = normalize_2d(proj)
            proj = proj.astype(np.float32)
            return proj, X,             scale

    def __len__(self):
        return len(self.data_list)


def normalize_3d(pose):  
    """
    pose: (length=1,17*3)
    Return:
        pose.T (length=1, 17*3)
    """      
    xs = pose.T[0::3] - pose.T[0] # (17*3, length)
    ys = pose.T[1::3] - pose.T[1]
    ls = np.sqrt(xs[1:] ** 2 + ys[1:] ** 2) 
    scale = ls.mean(axis=0) # (length, )
    pose = pose.T / scale
    pose[0::3] -= pose[0].copy()
    pose[1::3] -= pose[1].copy()
    pose[2::3] -= pose[2].copy()

    return pose.T, scale

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


def main():

    h36m = H36M('all', True, True) 
    train_loader = data.DataLoader(h36m, batch_size=1,shuffle=False)

    for batch_idx, (xy, X, scale) in enumerate(train_loader):
        
        print(xy.shape, X.shape)
        break


if __name__ == "__main__":
    
    # with open('./data/points_3d.pkl', 'rb') as f:
    #     p3d = pkl.load(f)
    # print(p3d['S1'].keys())
    # with open('./data/cameras.pkl', 'rb') as f:
    #     cams = pkl.load(f)
    # print(cams['S1'].keys())
    # with open('./data/sh_detect_2d.pkl', 'rb') as f:
    #     p2d_sh = pkl.load(f) 
    # print(p2d_sh['S1']['Directions']['54138969'].shape)

    main()
    # pass
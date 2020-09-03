import argparse
import time
import numpy as np
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

from utils.logger import Logger
from utils.misc import AverageMeter, mkdir_p, save_checkpoint, str_to_bool
from utils.vis import vis_3d_skeleton

from models.model import Lifter, Discriminator, Discriminator_wgan
from datasets.H36M import H36M


parser = argparse.ArgumentParser(description='PyTorch Training Human3.6M')

# Optimization options 
parser.add_argument('--epoches', default=50001, type=int, metavar='N')
parser.add_argument('--lift_lr', default=1e-4, type=float, metavar='LR',
                    help='learning rate for lifter')
parser.add_argument('--disc_lr', default=1e-4, type=float, metavar='LR',
                    help='learning rate for disc')

# Checkpoints
parser.add_argument('-c', '--checkpoint', default='/home/lyuheng/vision/unsupervised_3d_pose_lift/checkpoints',
                    help='path to save checkpoints')
parser.add_argument('--checkpoint_save_interval', default=5000, type=int, 
                    help='checkpoint save interval')
# Loss options
parser.add_argument('--weight_2d', default=10, type=float, 
                    help = 'loss weigth for 2d')
parser.add_argument('--weight_3d', default=0.001, type=float, 
                    help = 'loss weigth for 3d')
parser.add_argument('--weight_wt', default=1, type=float, 
                    help = 'loss weigth for temporal')
parser.add_argument('--lamda_gp', default=10, type=float,
                    help = 'lamda for gradient penalty')
# Model Params
parser.add_argument('--length', default=5, type=int,
                    help= 'temporal length')
# Training
parser.add_argument('--verbose', default=False, type=str_to_bool, nargs='?',
                    help = 'show detailed loss')
# Evaluation
parser.add_argument('--eval_interval', default=1000, type=int, 
                    help='evaluation interval')
parser.add_argument('--eval_dir', default='/home/lyuheng/vision/unsupervised_3d_pose_lift/eval2',
                    help='path to evaluation')

args = parser.parse_args()


def set_requires_grad(net, switch):
    for param in net.paramemters():
        param.requires_grad = switch
    return

def load_model(L, D, T, path):
    state = torch.load(path)
    L.load_state_dict(state['state_dict_L'])
    D.load_state_dict(state['state_dict_D'])
    T.load_state_dict(state['state_dict_T'])


def compute_gradient_penalty(D, xy_real, xy_fake, lamda, device):
    batch_sz = xy_real.size(0)
    alpha = torch.rand(batch_sz, 1)
    alpha = alpha.to(device)

    interpolates = alpha * xy_real + (1-alpha)*xy_fake
    interpolates = interpolates.to(device)

    interpolates = Variable(interpolates, requires_grad=True)

    interpolates_disc = D(interpolates)

    gradients = torch.autograd.grad(outputs=interpolates_disc, inputs=interpolates,
                                    grad_outputs=torch.ones(interpolates_disc.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = lamda*((gradients.norm(2,dim=1) - 1)**2).mean()

    return gradient_penalty
    


def main():

    if not osp.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    L = Lifter().to(device)
    D = Discriminator_wgan().to(device)
    T = Discriminator_wgan().to(device)

    optim_L = optim.Adam(L.parameters(), lr=args.lift_lr)
    optim_D = optim.Adam(D.parameters(), lr=args.disc_lr)
    optim_T = optim.Adam(T.parameters(), lr=args.disc_lr)

    # use 2D results from Stack Hourglass Net
    train_loader = data.DataLoader(
        H36M(length=args.length, action='all', is_train=True, use_sh_detection=True),
        batch_size=1024,
        shuffle=True,
        pin_memory=True,
    )

    test_loader = data.DataLoader(
        H36M(length=1, action='all', is_train=False, use_sh_detection=True),
        batch_size=512,
        shuffle=False,
    )

    # Logger
    logger = Logger(osp.join(args.checkpoint, 'log.txt'), title='Human3.6M')
    logger_err = Logger(osp.join(args.checkpoint, 'log_err.txt'), title='Human3.6M MPJPE err')
    logger.set_names(['2d_loss   ', '3d_loss   ', 'adv_loss   ', 'temporal_loss   '])
    logger_err.set_names(['err'])

    for epoch in range(args.epoches):

        print('\nEpoch: [%d / %d]' % (epoch+1, args.epoches))

        loss_2d, loss_3d, loss_adv, loss_t = train(train_loader, L, D, T, optim_L, optim_D, optim_T, epoch+1, device, args)

        logger.append([loss_2d, loss_3d, loss_adv, loss_t])
        
        if (epoch + 1) % args.checkpoint_save_interval == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict_L': L.state_dict(),
                'state_dict_D': D.state_dict(),
                'state_dict_T': T.state_dict(), 
            }, checkpoint=args.checkpoint)
        
        if (epoch + 1) % args.eval_interval == 0:
            ttl_err = test(test_loader, L, epoch, device, args)
            logger_err.append([ttl_err])

    logger.close()
    logger_err.close()



def train(train_loader, L, D, T, optim_L, optim_D, optim_T, epoch, device, args):

    L.train()
    D.train()
    T.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    loss_2ds = AverageMeter()
    loss_3ds = AverageMeter()
    loss_advs = AverageMeter()
    loss_ts = AverageMeter()

    end = time.time()

    L2_loss = nn.MSELoss(reduction='mean')

    for batch_idx, (xy, X, scale) in enumerate(train_loader):

        data_time.update(time.time() - end)
        # train D
        D.zero_grad()
        batch_sz = xy.size(0)
        pose_2d = xy[:, 0].to(device)  # (bs, 17*2)
        xy_real = xy[:, 1:].to(device) # (bs, length-1,17*2)

        z_pred = L(pose_2d) # (bs,17)

        # random Rotation
        theta = np.random.uniform(-np.pi, np.pi, batch_sz).astype(np.float32)
        cos_theta = np.cos(theta)[:, None]
        sin_theta = np.sin(theta)[:, None]

        cos_theta = torch.from_numpy(cos_theta).to(device)
        sin_theta = torch.from_numpy(sin_theta).to(device)

        x = pose_2d[:, 0::2]
        y = pose_2d[:, 1::2]
        new_x = x*cos_theta + z_pred*sin_theta # create projection

        trans_3d_z = -x*sin_theta + z_pred*cos_theta

        xy_fake = torch.cat((new_x[:,:,None], y[:,:,None]), dim=2)
        xy_fake = xy_fake.view(batch_sz, -1) # (bs, 17*2)

        trans_3d_1 = torch.cat((new_x[:,:,None], y[:,:,None], trans_3d_z[:,:,None]), dim=2)
        trans_3d_1 = trans_3d_1.view(batch_sz, -1) # (bs,17*3)

        D_real_loss = D( xy_real.view(batch_sz*(args.length-1), -1) ).mean() # (bs*(length-1),1)

        xy_fake_repeat = xy_fake.repeat(args.length-1, 1)
        D_fake_loss = D(xy_fake_repeat).mean()

        gradient_penalty_D = compute_gradient_penalty(D, xy_real.view(batch_sz*(args.length-1), -1), 
                                                        xy_fake_repeat, args.lamda_gp, device)
        
        D_train_loss_gp = D_fake_loss - D_real_loss + gradient_penalty_D
        D_train_loss_gp.backward(retain_graph=True)
        optim_D.step()

        # Train T
        T.zero_grad()
        pose_next = xy[:, 1].to(device) # (bs, 17*2)

        # Predict next pose
        z_pred_next = L(pose_next) # (bs,17)
        x_next = pose_next[:, 0::2]
        y_next = pose_next[:, 1::2]
        new_x_next = x_next*cos_theta + z_pred_next*sin_theta
        xy_fake_next = torch.cat((new_x_next[:,:, None], y_next[:,:, None]), dim=2)
        xy_fake_next = xy_fake_next.view(batch_sz,-1) # (bs, 17*2)

        T_real_loss = T(pose_2d - pose_next).mean()
        T_fake_loss = T(xy_fake - xy_fake_next).mean()
        gradient_penalty_T = compute_gradient_penalty(T, pose_2d-pose_next, xy_fake-xy_fake_next, 
                                                        args.lamda_gp, device)

        T_train_loss_gp = T_fake_loss - T_real_loss + gradient_penalty_T
        T_train_loss_gp.backward(retain_graph=True)
        optim_T.step()
        
        # Train L
        L.zero_grad()
        z_pred_fake_3d = L(xy_fake) # (bs,17)
        
        # Inverse 3D Transformation
        cos_theta_inv = np.cos(-theta)[:, None]
        sin_theta_inv = np.sin(-theta)[:, None]

        cos_theta_inv = torch.from_numpy(cos_theta_inv).to(device)
        sin_theta_inv = torch.from_numpy(sin_theta_inv).to(device)

        x_fake = xy_fake[:, 0::2]
        y_fake = xy_fake[:, 1::2]
        recover_new_x = x_fake*cos_theta_inv + z_pred_fake_3d*sin_theta_inv
        recover_xy = torch.cat((recover_new_x[:,:,None], y_fake[:,:,None]), dim=2)
        recover_xy = recover_xy.view(batch_sz, -1) # (bs, 17*2)

        trans_3d_2 = torch.cat((x_fake[:,:,None], y_fake[:,:,None], z_pred_fake_3d[:,:,None]), dim=2)
        trans_3d_2 = trans_3d_2.view(batch_sz, -1)
        
        loss_2d = L2_loss(recover_xy, pose_2d)
        loss_3d = L2_loss(trans_3d_1, trans_3d_2)

        loss_adv = -D(xy_fake).mean()

        loss_t = -T(xy_fake - xy_fake_next).mean()

        L_train_loss = loss_adv + args.weight_2d*loss_2d + args.weight_3d*loss_3d + args.weight_wt*loss_t

        L_train_loss.backward(retain_graph=True)
        optim_L.step()
        
        loss_2ds.update(loss_2d.item(), batch_sz)
        loss_3ds.update(loss_3d.item(), batch_sz)
        loss_advs.update(loss_adv.item(), batch_sz)
        loss_ts.update(loss_t.item(), batch_sz)

        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose:
            if batch_idx % 5 == 0:
                outstr = '[{batch}/{size}], Data: {data:.3f}s | Batch: {bt:.3f}s | loss_2d: {l2d:.6f} | loss_3d: {l3d:.6f} | loss_adv: {ladv:.6f} | loss_t: {lt:.6f}'.format(
                    batch = batch_idx + 1,
                    size = len(train_loader),
                    data = data_time.val,
                    bt = batch_time.val,
                    l2d = loss_2ds.val,
                    l3d = loss_3ds.val,
                    ladv = loss_advs.val,
                    lt = loss_ts.val,
                )
                print(outstr)

        return loss_2ds.avg, loss_3ds.avg, loss_advs.avg, loss_ts.avg

# TODO: 
def evaluate():
    pass

def test(test_loader, L, epoch, device, args):
    L.eval()
    if not osp.isdir(args.eval_dir):
        mkdir_p(args.eval_dir)

    all_dist = []
    for batch_idx, (xy, X, ls) in enumerate(test_loader):
        
        bs = xy.shape[0]
        xy = xy.squeeze(1) # (BS,17*2)
        xy = xy.to(device)

        z_pred = L(xy) # (bs,17)

        pose_3d_t = torch.cat(( xy[:,0::2,None], \
                                xy[:,1::2,None], \
                                z_pred[:,:,None]), dim=2)

        pose_3d_t = pose_3d_t.view(-1, 17*3).cpu().detach().numpy()
        
        X = X.squeeze(1).numpy()
        ls = ls.squeeze(1).numpy()

        pose_3d_t[:,0::3] = pose_3d_t[:,0::3] - pose_3d_t[:,0][:,None]
        pose_3d_t[:,1::3] = pose_3d_t[:,1::3] - pose_3d_t[:,1][:,None]
        pose_3d_t[:,2::3] = pose_3d_t[:,2::3] - pose_3d_t[:,2][:,None]

        if batch_idx == 0:
            vis_3d_skeleton(pose_3d_t[0].reshape(17,3), np.ones((17,1)), epoch=epoch+1)
        
        # use Protocal-1
        # for ba in range(bs):
        #     gt = X[ba].reshape(-1,3)
        #     out = pose_3d_t[ba].reshape(-1,3)
        #     _, Z, T, b, c = get_transformation(gt,out,True)
        #     out = (b*out.dot(T)) + c
        #     pose_3d_t[ba, :] = out.reshape(51)

        sqerr = (pose_3d_t - X)**2

        distance = np.zeros((bs, 17))
        dist_idx = 0
        for k in range(0, 17*3, 3):
            distance[:, dist_idx] = np.sqrt(np.sum(sqerr[:,k:k+3],axis=1))*ls
            dist_idx += 1

        all_dist.append(distance)
    
    all_dist = np.vstack(all_dist)
    ttl_err = np.mean(all_dist)

    return ttl_err


def get_transformation(X, Y, compute_optimal_scale=False):
    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:, -1] *= np.sign(detT)
    s[-1] *= np.sign(detT)
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - traceTA ** 2
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    c = muX - b * np.dot(muY, T)

    return d, Z, T, b, c


if __name__ == "__main__":
    main()
        
        
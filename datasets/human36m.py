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

"""
8.11 test data from points_3d.pkl
"""


skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), 
            (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )

def world2cam(world_coord, R, T):
    """
    Args:
        world_coord: (17,3)
        R: (3,3)
        T: (3,1)
    return:
        cam_coord: (17,3)
    """
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3) # changed
    return cam_coord


def cam2pixel(cam_coord, f, c):
    """

    return:
        img_coord: (17,3)
    """
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord


def process_bbox(bbox, width, height):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
    if w*h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2-x1, y2-y1])
    else:
        return None

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = 1    # cfg.input_shape[1]/cfg.input_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    return bbox



def vis_frame(joints, bbox, parents, joints_right, img, colors=None,
              use_score=False, joint_scores=None, threshold=None):
    """Draw joints in original image

    Args:
        joints: joint position, np.array, (n, 2)
        parents: joint parent id, np.array, (n, )
        joints_right: right joint ids, np.array, (m, )
        img: original image, np.array, (height, width, 3) # (1000,1000,3)
        threshold: threshold for drawing joints
        colors: color is used to draw joints and bones, list, [(255, 0, 0), ...]
        joint_scores: confidence of estimated joint, np.array, (n, 3)
        use_score: if True, alpha = score

    Returns:
        vis_img: drawing result, np.array, (height, width, 4)
    """
    if use_score is True:
        assert joint_scores is not None
        assert threshold is not None

    if threshold is None:
        threshold = 0.0

    if joint_scores is None:
        num_joints = joints.shape[0]
        joint_scores = np.ones(num_joints)

    if colors is None:
        colors = [(255, 0, 0) if j in joints_right else (0, 0, 255) for j in range(joints.shape[0])]
    
    # check data from points_3d.pkl
    """
    with open('./data/sh_detect_2d.pkl', 'rb') as f:
        p2d_sh = pkl.load(f)

    joints = np.array(p2d_sh['S1']['Directions 1']['54138969'][0])
    joints = joints.reshape(-1,2)
    """
    
    for j in range(joints.shape[0]):
        if joint_scores[j] < threshold:
            continue
        x, y = int(joints[j, 0]), int(joints[j, 1])
        cv2.circle(img, (x, y), 6, colors[j], -1)

        j_parent = parents[j]
        if j_parent < 0 or joint_scores[j_parent] < threshold:
            continue

        x2, y2 = int(joints[j_parent, 0]), int(joints[j_parent, 1])
        cv2.line(img, (x, y), (x2, y2), colors[j], thickness=3)  # Control thickness with score

    x,y,w,h = bbox
    cv2.rectangle(img, (int(x),int(y)), (int(x+w), int(y+h)), color=(0,255,255), thickness=3)  # 造谣!


    x,y,w,h = process_bbox(bbox, 1000, 1000)

    cv2.rectangle(img, (int(x),int(y)), (int(x+w), int(y+h)), color=(0, 255, 0), thickness=3)  # 造谣!
    return img



def generate_patch_image(cvimg, bbox):
    img = cvimg.copy()
    img_h, img_w, img_c = img.shape
    bb_c_x = float(bbox[0]+0.5*bbox[2])
    bb_c_y = float(bbox[1]+0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, 256, 256)

    img_patch = cv2.warpAffine(img, trans, (256, 256), flags=cv2.INTER_LINEAR)

    img_patch = img_patch[:,:,::-1].copy()
    img_patch = img_patch.astype(np.float32)

    return img_patch, trans




def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale=1.0, rot=0.0,   inv=False):
    """
    port from 
    https://github.com/mks0601/3DMPPE_POSENET_RELEASE/blob/master/data/dataset.py

    """
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T # point
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]


def render_2d(input_img_dir, seq_2d, joint_parent, joint_right, img_dir, data, joint_imgs):
    if osp.exists(img_dir):
        shutil.rmtree(img_dir)

    base_name = osp.basename(input_img_dir)
    print(f'Render 2d pose to {img_dir}.')
    print(f'Human 36m inputs dir {base_name}')
    os.makedirs(img_dir, exist_ok=True)
    for frame_id in range(0, seq_2d.shape[0], 50):
        pose_2d = seq_2d[frame_id]
        joint_img = joint_imgs[frame_id]

        img_file = osp.join(img_dir, f'frame_{frame_id}.jpg')

        # img = np.ones((height, width, 3), dtype=np.uint8) * 255
        input_img_file = f'{input_img_dir}/{base_name}_{(frame_id+1):06}.jpg'  # human36m start from 1

        data_file_name = input_img_file.split('/')[-2:]
        data_file_name = data_file_name[0] + '/' + data_file_name[1]
        
        idx_cnt = 0
        for idx, d in enumerate(data['images']):
            if d['file_name'] == data_file_name:
                idx_cnt = d['id']
                break
        
        assert idx_cnt == data['annotations'][idx_cnt]['id']
        bbox = data['annotations'][idx_cnt]['bbox']
        
        assert osp.exists(input_img_file)
        img = cv2.imread(input_img_file)
        vis_img = vis_frame(pose_2d, bbox, joint_parent, joint_right, img)
        cv2.imwrite(img_file, vis_img)

        img_patch, trans = generate_patch_image(img, bbox) # no augmentation at all 

        for i in range(len(joint_img)):
            joint_img[i, 0:2] = trans_point2d(joint_img[i, 0:2], trans)
            joint_img[i, 2] /= (2000/2.)
            joint_img[i, 2] = (joint_img[i, 2] + 1.0)/2.

        # print(joint_img)
        vis_3d_skeleton(joint_img, np.ones((17,1)), skeleton)


def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

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
    ax.set_xlim([0,256])
    ax.set_ylim([0,1])
    ax.set_zlim([-256,0])
    ax.legend()

    plt.show()
    cv2.waitKey(0)


if __name__ == "__main__":
    base_dir = '/media/lyuheng/Seagate Portable Drive'
    h36m_img_dir = osp.join(base_dir, 'Human36m', 'images')
    annot_path = osp.join(base_dir, 'Human36m', 'annotations')

    subject_id = 1
    act_id = 2
    sub_act_id = 1
    h36m_cam_id = 1

    joints_file = osp.join(annot_path, 'Human36M_subject' + str(subject_id) + '_joint_3d.json')
    assert osp.exists(joints_file)
    with open(joints_file, 'r') as f:
        joints = json.load(f)

    camera_info_file = osp.join(annot_path, 'Human36M_subject' + str(subject_id) + '_camera.json')
    with open(camera_info_file, 'r') as f:
        cameras = json.load(f)

    data_info_file = osp.join(annot_path, 'Human36M_subject' + str(subject_id) + '_data.json')
    with open(data_info_file, 'r') as f:
        data = json.load(f)

    cam_param = cameras[str(h36m_cam_id)]   
    R = np.array(cam_param['R'], dtype=np.float32)
    t = np.array(cam_param['t'], dtype=np.float32)
    f = np.array(cam_param['f'], dtype=np.float32)
    c = np.array(cam_param['c'], dtype=np.float32)

    joints_act = joints[str(act_id)][str(sub_act_id)]


    #print(joints_act.keys())

    duration = len(joints_act.keys())
    seq_pose = []
    seq_pose_2d = []
    joint_imgs = []

    print(duration)

    for frame_id in range(duration):
        pose_3d_frame = np.array(joints_act[str(frame_id)], dtype=np.float32)
        # print(pose_3d_frame)
        seq_pose.append(pose_3d_frame)

        # 2d pose
        # print(pose_3d_frame.shape) 
        # print(t.shape) (3,)
        pose_3d_frame_cam = world2cam(pose_3d_frame, R, t.reshape(3, 1))
        pose_2d_img = cam2pixel(pose_3d_frame_cam, f, c)

        joint_img = pose_2d_img.copy()

        pose_2d_img = pose_2d_img[:, :2]
        seq_pose_2d.append(pose_2d_img)

        joint_img[:,2] = joint_img[:,2]-pose_3d_frame_cam[0,2] # Pelvis joint

        joint_imgs.append(joint_img)
        break

    
    seq_pose_2d = np.array(seq_pose_2d, dtype=np.float32)
    joint_imgs = np.array(joint_imgs, dtype=np.float32)

    joint_parent = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    joint_right = [1, 2, 3, 14, 15, 16]

    subject_dir = './S9'
    os.makedirs(subject_dir, exist_ok=True)
    input_img_dir = f'{h36m_img_dir}/' \
                    f's_{subject_id:02}' \
                    f'_act_{act_id:02}' \
                    f'_subact_{sub_act_id:02}' \
                    f'_ca_{h36m_cam_id:02}'
    print(f'Human36m pose corresponding image dir {input_img_dir}.')
    img_dir = osp.join(subject_dir, f'act_{act_id}_subact_{sub_act_id}')
    video_file = osp.join(subject_dir, f's_{subject_id:02}_act_{act_id}_subact_{sub_act_id}_cam{h36m_cam_id}_2d.mp4')
    gif_file = osp.join(subject_dir, f's_{subject_id:02}_act_{act_id}_subact_{sub_act_id}_cam{h36m_cam_id}_2d.gif')
    render_2d(input_img_dir,
              seq_pose_2d,
              joint_parent,
              joint_right,
              img_dir,
              data,
              joint_imgs
            )
    

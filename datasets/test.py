
import os
import os.path as osp
import numpy as np
import json
import cv2
import shutil

# base_dir = '/media/lyuheng/Seagate Portable Drive'
# h36m_img_dir = osp.join(base_dir, 'Human36m', 'images')
# annot_path = osp.join(base_dir, 'Human36m', 'annotations')

# subject_id = 1
# act_id = 2
# sub_act_id = 1
# h36m_cam_id = 3

# joints_file = osp.join(annot_path, 'Human36M_subject' + str(subject_id) + '_joint_3d.json')
# data_file = osp.join(annot_path, 'Human36M_subject' + str(subject_id) + '_data.json')
# assert osp.exists(data_file)
# with open(data_file, 'r') as f:
#     data = json.load(f)

# print(len(data['images']))
"""
{'id': 0, 
'file_name': 's_01_act_02_subact_01_ca_01/s_01_act_02_subact_01_ca_01_000001.jpg', 
'width': 1000, 
'height': 1002, 
'subject': 1, 
'action_name': 'Directions', 
'action_idx': 2, 
'subaction_idx': 1, 
'cam_idx': 1, 
'frame_idx': 0}
"""

# print(len(data['annotations']))

"""
{'id': 0, 
'image_id': 0, 
'keypoints_vis': [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True], 
'bbox': [402.65667804648615, 262.87433777970045, 127.8323473589618, 404.42892185240856]}
"""


a = [1,2,3]
print(a[0::3])
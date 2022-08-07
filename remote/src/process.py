# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from HybrIK.hybrik.models import builder
from HybrIK.hybrik.utils.config import update_config
from HybrIK.hybrik.utils.presets import SimpleTransform3DSMPL
from HybrIK.hybrik.utils.render import SMPLRenderer
from HybrIK.hybrik.utils.vis import get_one_box, vis_smpl_3d
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm
from HybrIK.hybrik.utils.render import draw_skeleton

# Ref 
# https://github.com/Jeff-sjtu/HybrIK
def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]

class Params:
  def __init__(self):
    self.gpu = 7
    self.out_dir = "res"
    self.img_dir = "data"


def get_joints():
    """
    画像から得た3D空間上の座標を返す
    Returns :
        joint2 (numpy.array) : 画像から得た3D空間上の座標 
    """
    det_transform = T.Compose([T.ToTensor()])
    CKPT = './HybrIK/pretrained_w_cam.pth'
    opt = Params()

    cfg_file = './HybrIK/configs/256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix.yaml'
    cfg = update_config(cfg_file)
    dummpy_set = edict({
    	'joint_pairs_17': None,
    	'joint_pairs_24': None,
    	'joint_pairs_29': None,
    	'bbox_3d_shape': (2.2, 2.2, 2.2)
    })

    transformation = SimpleTransform3DSMPL(
    	dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
    	color_factor=cfg.DATASET.COLOR_FACTOR,
    	occlusion=cfg.DATASET.OCCLUSION,
    	input_size=cfg.MODEL.IMAGE_SIZE,
    	output_size=cfg.MODEL.HEATMAP_SIZE,
    	depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
    	bbox_3d_shape=(2.2, 2,2, 2.2),
    	rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
    	train=False, add_dpg=False,
    	loss_type=cfg.LOSS['TYPE'])
    det_model = fasterrcnn_resnet50_fpn(pretrained=True)
    hybrik_model = builder.build_sppe(cfg.MODEL)

    print(f'Loading model from {CKPT}...')
    hybrik_model.load_state_dict(torch.load(CKPT, map_location='cpu'), strict=False)

    det_model.cuda(opt.gpu)
    hybrik_model.cuda(opt.gpu)
    det_model.eval()
    hybrik_model.eval()
    files = os.listdir(opt.img_dir)

    if not os.path.exists(opt.out_dir):
    	os.makedirs(opt.out_dir)
    
    pose_input_list  =[] #
    bbox_list = [] #

    _file =""
    for file in tqdm(files):
    	if not os.path.isdir(file) and file[-4:] in ['.jpg', '.png']:
    		# is an image
    		if file[:4] == 'res_':
    			continue
    		_file = file

    file = _file
    print(file)
    # process file name
    img_path = os.path.join(opt.img_dir, file)
    dirname = os.path.dirname(img_path)
    basename = os.path.basename(img_path)

    # Run Detection
    input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    det_input = det_transform(input_image).to(opt.gpu)
    det_output = det_model([det_input])[0]

    tight_bbox = get_one_box(det_output)  # xyxy

    # Run HybrIK
    pose_input, bbox = transformation.test_transform(img_path, tight_bbox)
    pose_input_list.append(pose_input) #
    bbox_list.append(bbox) #

    pose_input = pose_input.to(opt.gpu)[None, :, :, :]
    pose_output = hybrik_model(pose_input)

    pose_out=pose_output.pred_xyz_jts_17.detach().cpu().numpy()
    pose_out_=np.reshape(pose_out,(17,3))

    # 今回必要であった19個のjointsを取り出す
    joint = []
    joint.append(pose_out_[6]) #0
    joint.append(pose_out_[5]) #1
    joint.append(pose_out_[4]) #2
    joint.append(pose_out_[1]) #3
    joint.append(pose_out_[2]) #4
    joint.append(pose_out_[3]) #5
    joint.append(pose_out_[16]) #6
    joint.append(pose_out_[15]) #7
    joint.append(pose_out_[14]) #8
    joint.append(pose_out_[11]) #9
    joint.append(pose_out_[12]) #10
    joint.append(pose_out_[13]) #11
    joint.append(pose_out_[8]) #12
    joint.append(pose_out_[10]) #13
    joint.append(pose_out_[9]) #14
    joint.append(pose_out_[9]) #15
    joint.append(pose_out_[9]) #16
    joint.append(pose_out_[9]) #17
    joint.append(pose_out_[9]) #18

    print("joints {}".format(len(joint)))
    joint2 = np.array(joint)
    print(joint2.shape)
    return joint2
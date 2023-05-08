import os, json
from os import path as osp
import torch
import numpy as np
import imageio 
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w



def load_bop_linemod_data(basedir, half_res=False, obj='000001', normalize_factor=1.):
    # load object pose 
    pose_json = osp.join(basedir, obj, 'scene_gt.json')
    with open(pose_json, 'r') as f:
        annotations = json.load(f)
        all_TCO = []
        for image_id in annotations:
            RCO = np.array(annotations[image_id][0]['cam_R_m2c'], dtype=np.float32).reshape(3,3)
            tCO = np.array(annotations[image_id][0]['cam_t_m2c'], dtype=np.float32).t()
            TCO = np.zeros((4, 4), dtype=np.float32) / normalize_factor
            TCO[:3,:3] = RCO
            TCO[:3, -1] = tCO
            TCO[-1, -1] = 1. 
            all_TCO.apped(TCO)
    
    # load images and image lists
    splits = ['train', 'val', 'test']
    image_lists = {}
    all_images = []
    for s in splits:
        with open(os.path.join(basedir, 'nerf_image_lists', obj+'_'+s+'.txt'), 'r') as f:
            image_lists[s] = list(map(lambda x:x.strip(), f.readlines()))
        for image_path in image_lists[s]:
            image = imageio.imread(osp.join(basedir, obj, 'rgb', image_path)) #RGB
            mask = cv2.imread(osp.join(basedir, 'mask_visib', image_path.split('.')[0]+'_000000.png'))
            image = (np.array(image) / 255).astype(np.float32)
            image = np.concatenate([image, (mask/255).astype(np.float32)], axis=0) # RGBA
            all_images.append(image)
    
    all_images = np.stack(all_images, axis=0)
    all_TCO = np.stack(all_TCO, axis=0)
    all_TOC = np.linalg.inv(all_TCO) # object pose -> camera pose
    
    # reformat poses
    new_TOC, i_split = [], []
    count = 0
    for s in image_lists:
        image_index = np.array(list(map(lambda x:int(x.split('.')[0]), image_lists[s])), dtype=np.int64)
        new_TOC.append(all_TOC[image_index])
        i_split.append(np.arange(count, count+len(image_index)))
        count += len(image_index)
    all_TOC = np.concatenate(new_TOC, axis=0)

    camera_k = np.array([[572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0]], dtype=np.float32).reshape(3, 3)
    H, W = 480, 640

    render_poses = torch.stack([pose_spherical(angle, -30.0, 12.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    if half_res:
        H, W = H//2, W//2
        camera_k = camera_k / 2
        images_half_res = np.zeros((all_images.shape[0], H, W, 4))
        for i, img in enumerate(all_images):
            images_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        all_images = images_half_res
        
    return all_images, all_TOC, render_poses, camera_k, i_split

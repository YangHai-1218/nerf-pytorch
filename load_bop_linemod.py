import os, json
from os import path as osp
import torch, torchvision
import numpy as np
import imageio 
import cv2
from tqdm import tqdm


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


def project_3d_point(pt3d, K, rotation, translation, return_3d=False):
    pt_3d_camera = np.matmul(rotation, pt3d.T) + translation
    pt_2d = np.matmul(K, pt_3d_camera).T
    pt_2d[..., 0] = pt_2d[..., 0]/ (pt_2d[..., -1] + 1e-8)
    pt_2d[..., 1] = pt_2d[..., 1]/ (pt_2d[..., -1] + 1e-8)
    pt_2d = pt_2d[..., :-1]

    if return_3d:
        return pt_2d, pt_3d_camera.T
    else:
        return pt_2d

def deepim_bbox(uv_center, obs_bboxes, ):
    pass
    


def get_K_crop_resize(K, boxes, orig_size, crop_resize):
    """
    Adapted from https://github.com/BerkeleyAutomation/perception/blob/master/perception/camera_intrinsics.py
    Skew is not handled !
    """
    assert K.shape[1:] == (3, 3)
    assert boxes.shape[1:] == (4, )
    K = K.float()
    boxes = boxes.float()
    new_K = K.clone()

    orig_size = torch.as_tensor(orig_size, dtype=torch.float).clone()
    crop_resize = torch.as_tensor(crop_resize, dtype=torch.float).clone()

    final_width, final_height = max(crop_resize), min(crop_resize)
    crop_width = boxes[:, 2] - boxes[:, 0]
    crop_height = boxes[:, 3] - boxes[:, 1]
    crop_cj = (boxes[:, 0] + boxes[:, 2]) / 2
    crop_ci = (boxes[:, 1] + boxes[:, 3]) / 2

    # Crop
    cx = K[:, 0, 2] + (crop_width - 1) / 2 - crop_cj
    cy = K[:, 1, 2] + (crop_height - 1) / 2 - crop_ci

    # # Resize (upsample)
    center_x = (crop_width - 1) / 2
    center_y = (crop_height - 1) / 2
    orig_cx_diff = cx - center_x
    orig_cy_diff = cy - center_y
    scale_x = final_width / crop_width
    scale_y = final_height / crop_height
    scaled_center_x = (final_width - 1) / 2
    scaled_center_y = (final_height - 1) / 2
    fx = scale_x * K[:, 0, 0]
    fy = scale_y * K[:, 1, 1]
    cx = scaled_center_x + scale_x * orig_cx_diff
    cy = scaled_center_y + scale_y * orig_cy_diff

    new_K[:, 0, 0] = fx
    new_K[:, 1, 1] = fy
    new_K[:, 0, 2] = cx
    new_K[:, 1, 2] = cy
    return new_K

def crop_and_resize(bbox, image, target_h, target_w, camera_k, TCO, mask=None):
    orig_h, orig_w = image.shape[-2:]
    uv_center = project_3d_point(np.array([0, 0, 0], dtype=np.float32), camera_k, TCO[:3, :3], TCO[:3, -1])


    



def load_bop_linemod_data(basedir, half_res=False, obj='000001', normalize_factor=1.):
    # load object pose 
    pose_json = osp.join(basedir, obj, 'scene_gt.json')
    with open(pose_json, 'r') as f:
        annotations = json.load(f)
        all_TCO = []
        for image_id in annotations:
            RCO = np.array(annotations[image_id][0]['cam_R_m2c'], dtype=np.float32).reshape(3,3)
            tCO = np.array(annotations[image_id][0]['cam_t_m2c'], dtype=np.float32).T / normalize_factor
            TCO = np.zeros((4, 4), dtype=np.float32)
            TCO[:3,:3] = RCO
            TCO[:3, -1] = tCO
            TCO[-1, -1] = 1. 
            all_TCO.append(TCO)
    
    # load images and image lists
    splits = ['train', 'val', 'test']
    image_lists = {}
    all_images = []
    for s in splits:
        with open(osp.join(osp.dirname(basedir), 'nerf_image_lists', obj+'_'+s+'.txt'), 'r') as f:
            image_lists[s] = list(map(lambda x:x.strip(), f.readlines()))
        pbar = tqdm(image_lists[s])
        for image_path in pbar:
            image = imageio.imread(osp.join(basedir, obj, 'rgb', image_path)) #RGB
            mask = imageio.imread(osp.join(basedir, obj, 'mask_visib', image_path.split('.')[0]+'_000000.png'))
            image = (image / 255).astype(np.float32)
            mask = (mask / 255).astype(np.float32)
            image = np.concatenate([image, mask[..., None]], axis=-1) # RGBA
            all_images.append(image)
    
    all_images = np.stack(all_images, axis=0)
    all_TCO = np.stack(all_TCO, axis=0)
    
    # reformat poses
    new_TCO, i_split = [], []
    count = 0
    for s in image_lists:
        image_index = np.array(list(map(lambda x:int(x.split('.')[0]), image_lists[s])), dtype=np.int64)
        new_TCO.append(all_TCO[image_index])
        i_split.append(np.arange(count, count+len(image_index)))
        count += len(image_index)
    all_TCO = np.concatenate(new_TCO, axis=0)
    all_TOC = np.linalg.inv(all_TCO) # object pose -> camera pose

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
        
    return all_images, all_TOC, render_poses, (H, W, camera_k), i_split

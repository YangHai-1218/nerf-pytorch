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

    
        


def get_center_offset(center, scale, ht, wd):
    upper = max(0, int(center[0] - scale / 2. + 0.5))
    left = max(0, int(center[1] - scale / 2. + 0.5))
    bottom = min(ht, int(center[0] - scale / 2. + 0.5) + int(scale))
    right = min(wd, int(center[1] - scale / 2. + 0.5) + int(scale))

    if upper == 0:
        h_offset = - int(center[0] - scale / 2. + 0.5) / 2
    elif bottom == ht:
        h_offset = - (int(center[0] - scale / 2. + 0.5) + int(scale) - ht) / 2
    else:
        h_offset = 0

    if left == 0:
        w_offset = - int(center[1] - scale / 2. + 0.5) / 2
    elif right == wd:
        w_offset = - (int(center[1] - scale / 2. + 0.5) + int(scale) - wd) / 2
    else:
        w_offset = 0
    center_offset = np.array([h_offset, w_offset])
    return center_offset

def Crop_by_Pad(img, center, size, res=None, channel=3, interpolation=cv2.INTER_LINEAR, resize=True):
    # Code from CDPN
    ht, wd = img.shape[0], img.shape[1]

    upper = max(0, int(center[0] - size / 2. + 0.5))
    left = max(0, int(center[1] - size / 2. + 0.5))
    bottom = min(ht, int(center[0] - size / 2. + 0.5) + int(size))
    right = min(wd, int(center[1] - size / 2. + 0.5) + int(size))
    crop_ht = float(bottom - upper)
    crop_wd = float(right - left)

    if resize:
        if crop_ht > crop_wd:
            resize_ht = res
            resize_wd = int(res / crop_ht * crop_wd + 0.5)
        elif crop_ht < crop_wd:
            resize_wd = res
            resize_ht = int(res / crop_wd * crop_ht + 0.5)
        else:
            resize_wd = resize_ht = int(res)

    tmpImg = img[upper:bottom, left:right]
    if not resize:
        outImg = np.zeros((int(size), int(size), channel))
        outImg[int(size / 2.0 - (bottom - upper) / 2.0 + 0.5):(
                int(size / 2.0 - (bottom - upper) / 2.0 + 0.5) + (bottom - upper)),
        int(size / 2.0 - (right - left) / 2.0 + 0.5):(
                int(size / 2.0 - (right - left) / 2.0 + 0.5) + (right - left)), :] = tmpImg
        return outImg

    resizeImg = cv2.resize(tmpImg, (resize_wd, resize_ht), interpolation=interpolation)
    if len(resizeImg.shape) < 3:
        resizeImg = np.expand_dims(resizeImg, axis=-1)  # for depth image, add the third dimension
    outImg = np.zeros((res, res, channel))
    outImg[int(res / 2.0 - resize_ht / 2.0 + 0.5):(int(res / 2.0 - resize_ht / 2.0 + 0.5) + resize_ht),
    int(res / 2.0 - resize_wd / 2.0 + 0.5):(int(res / 2.0 - resize_wd / 2.0 + 0.5) + resize_wd), :] = resizeImg
    return outImg


def trasnform_camera_k(camera_k, resize_factor, crop_center, target_size):
    K = camera_k.copy()
    K[0, 0] = K[0, 0] * resize_factor
    K[1, 1] = K[1, 1] * resize_factor
    K[0, 2] = (K[0, 2] + 0.5) * resize_factor - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * resize_factor - 0.5

    top_left = crop_center * resize_factor - target_size / 2
    K[0, 2] = K[0, 2] - top_left[1]
    K[1, 2] = K[1, 2] - top_left[0]

    return K


def crop_and_resize(bbox, image, target_size, camera_k, scale, mask=None):
    orig_h, orig_w = image.shape[-3:-1]
    left, top, width, height = bbox
    center = np.array([int(top + height / 2), int(left + width / 2)])
    size = int(scale * max(height, width))
    resize_factor = target_size / size 
    
    image = Crop_by_Pad(image, center, size, target_size, channel=3)
    center_offset = get_center_offset(center, size, orig_h, orig_w)
    camera_k = trasnform_camera_k(camera_k, resize_factor, center+center_offset, target_size)
    if mask is not None:
        mask = Crop_by_Pad(mask, center, size, target_size, channel=1)
        return image, camera_k, mask 
    else:
        return image, camera_k
    


    



def load_bop_linemod_data(basedir, half_res=False, obj='000001', normalize_factor=1., crop=False):
    # load object pose 
    pose_json = osp.join(basedir, obj, 'scene_gt.json')
    info_json = osp.join(basedir, obj, 'scene_gt_info.json')
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
    with open(info_json, 'r') as f:
        annotations = json.load(f)
        all_bboxes = []
        for image_id in annotations:
            bbox = np.array(annotations[image_id][0]['bbox_obj'], dtype=np.float32).reshape(-1)
            all_bboxes.append(bbox)
        all_bboxes = np.stack(all_bboxes).reshape(-1, 4)

    
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

    camera_k = np.array([[572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0]], dtype=np.float32).reshape(3, 3)
    H, W = 480, 640

    # reformat poses
    image_index_list, i_split = [], []
    count = 0
    for s in image_lists:
        image_index = np.array(list(map(lambda x:int(x.split('.')[0]), image_lists[s])), dtype=np.int64)
        image_index_list.append(image_index)
        i_split.append(np.arange(count, count+len(image_index)))
        count += len(image_index)
    
    image_index = np.concatenate(image_index_list)
    all_TCO = all_TCO[image_index]
    all_TOC = np.linalg.inv(all_TCO) # object pose -> camera pose
    all_bboxes = all_bboxes[image_index]
    
    if crop:
        target_size = 256
        H, W = target_size, target_size
        new_images, new_camera_k_list = [], []
        for i in range(len(all_images)):
            new_image, new_camera_k, new_mask = crop_and_resize(
                all_bboxes[i], all_images[i, ..., :3], target_size, camera_k, scale=1.4, mask=all_images[i, ..., -1])
            new_image = np.concatenate([new_image, new_mask], axis=-1)
            new_images.append(new_image)
            new_camera_k_list.append(new_camera_k)
        camera_k = np.stack(new_camera_k_list, axis=0)
        all_images = np.stack(new_images, axis=0)


    render_poses = torch.stack([pose_spherical(angle, -30.0, 12.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    if half_res:
        H, W = H//2, W//2
        camera_k = camera_k / 2
        images_half_res = np.zeros((all_images.shape[0], H, W, 4))
        for i, img in enumerate(all_images):
            images_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        all_images = images_half_res
        
    return all_images, all_TOC, render_poses, (H, W, camera_k), i_split

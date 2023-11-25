'''
File Created:  2023-Nov-13th Mon 10:28
Author:        Kaiyuan Wang (k5wang@ucsd.edu)
Affiliation:   ARCLab @ UCSD
Description:   Utilities for training, testing, visualization
'''


import torch, numpy as np, matplotlib.pyplot as plt, io
from PIL import Image
from kornia.geometry.liegroup import Se3
from kornia.geometry.quaternion import Quaternion

class AverageMeter:
    def __init__(self): self.n, self.avg = 0, 0.0

    def update(self, x):
        self.n += 1
        self.avg = ((self.n - 1) * self.avg + x) / self.n
        return self.avg

def to_device(*args, device='cuda' if torch.cuda.is_available() else 'cpu'):
    return [x.to(device) for x in args]

def to_numpy(*args):
    return [x.detach().cpu().numpy() for x in args]

def project_3d_2d(points_3d, K, pose_to_camera:Se3=None):
    ''' Project 3D points to 2D with optional camera pose
    Inputs:
        - points_3d: torch.Tensor of shape (N, 3) containing the 3D points
        - K: torch.Tensor of shape (3,3) containing camera intrinsics
        - pose_to_camera: kornia Se3 object.
        of the camera in the world coordinate frame (optional)
    '''
    if pose_to_camera is not None:  
        R = pose_to_camera.matrix()[..., :3,:3].squeeze().to(points_3d.device)
        t = pose_to_camera.matrix()[..., :3,3].squeeze().to(points_3d.device)
        points_3d = points_3d @ R.T + t

    points_2d_homo = points_3d @ K.T
    points_2d = points_2d_homo[:,:-1] / points_2d_homo[:,-1:]    # Dehemogenize
    return points_2d

def denormalize_img(img, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])):
    if isinstance(img, torch.Tensor): img = img.detach().cpu().numpy()
    img = img * std[None,:] + mean[None,:]
    return np.clip(img, 0,1)

def plot_keypoints_2d(kp2d_gt, kp2d_est, fname='keypoints2d.png'):
    '''Plot estimated 2d keypoints to ground truth
    Inputs:
        - kp2d_gt, kp2d_est: torch.Tensor of shape (N, 2) containing the ground truth and estimated 2D keypoints
    '''
    # Plot ground truth keypoints
    if isinstance(kp2d_gt, torch.Tensor): kp2d_gt = kp2d_gt.detach().cpu().numpy()
    if isinstance(kp2d_est, torch.Tensor): kp2d_est = kp2d_est.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10,10))
    ax.invert_yaxis()
    ax.scatter(kp2d_gt[:,0], kp2d_gt[:,1], color='b', label='Ground Truth (hand marked on image)')
    ax.scatter(kp2d_est[:,0], kp2d_est[:,1], color='r', label='Estimated using inverse of CtR Pose')

    for i in range(min(len(kp2d_est), len(kp2d_gt))):
        plt.text(kp2d_gt[i,0], kp2d_gt[i,1], str(i+1), ha='center', va='bottom')
        plt.text(kp2d_est[i,0], kp2d_est[i,1], str(i+1), ha='center', va='bottom')
    ax.legend(); plt.show(); plt.savefig(fname); plt.clf()

def get_bbox(points):
    ''' Get bounding box from points matrix of shape (N, 2) or (N, 3) 
    Returns:
        - vertices: torch.Tensor of shape (8, 3) containing the 8 vertices of 3d bounding box
                    or 
                    torch.Tensor of shape (4, 3) containing the 4 vertices of 2d bounding box

    Note: 3d vertex labling order:
                  4 ---- 7
                / |     /|
                5 ---- 6 |
                | 0 ---- 3
                | /    |/
                1 ---- 2
    '''
    assert points.shape[-1] in [2,3], f'Only support 2d or 3d points. But got points.shape[-1]={points.shape[-1]}'
    if points.shape[-1] == 3:
        xyz_min = torch.tensor([points[:,0].min(), points[:,1].min(), points[:,2].min()], 
                               dtype=points.dtype, device=points.device)
        xyz_max = torch.tensor([points[:,0].max(), points[:,1].max(), points[:,2].max()], 
                        dtype=points.dtype, device=points.device)
        vertices = torch.tensor([
            [xyz_min[0], xyz_min[1], xyz_min[2]],
            [xyz_min[0], xyz_min[1], xyz_max[2]],
            [xyz_min[0], xyz_max[1], xyz_max[2]],
            [xyz_min[0], xyz_max[1], xyz_min[2]],
            [xyz_max[0], xyz_min[1], xyz_min[2]],
            [xyz_max[0], xyz_min[1], xyz_max[2]],
            [xyz_max[0], xyz_max[1], xyz_max[2]],
            [xyz_max[0], xyz_max[1], xyz_min[2]],
        ], dtype=points.dtype, device=points.device)

        return vertices

    else: raise NotImplementedError # See dev_sam.py, kps_to_bbox()

def show_box(box, ax, type='3d', color='white'):
    ''' Overlay 3d or 2d bounding box on matplotlib axis 
    Input: 
        - box:  torch.Tensor (4, 3) containing the 4 vertices of 2d bounding box
                or 
                torch.Tensor (8, 3) containing the 8 vertices of 3d bounding box 
    '''
    assert type in ['3d', '2d']
    if isinstance(box, torch.Tensor): box = box.detach().cpu().numpy()

    if type == '2d':
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=2))    
    elif type == '3d':
        lines = np.array([
            [0,1], [1,2], [2,3], [3,0],
            [4,5], [5,6], [6,7], [7,4],
            [0,4], [1,5], [2,6], [3,7],
        ])
        for line in lines: ax.plot(box[line,0], box[line,1], color=color, linewidth=2)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.8])
    h, w = mask.shape[-2:]
    mask_image = (mask.reshape(h, w, 1) * color.reshape(1, 1, -1)).astype(np.uint8)
    ax.imshow(mask_image)

def show_points(points, ax, color='white', marker_size=100, label=None):
    ''' Overlay points on matplotlib axis'''
    if isinstance(points, torch.Tensor): points = points.detach().cpu().numpy()
    ax.scatter(points[:, 0], points[:, 1], color=color, marker='*', s=marker_size, label=label)

def plot_pose_and_gtkp(cTr, K, kp3d, kp2d_est, kp2d_gt, img, seg, fname=None, title=None):
    '''Plot estimated 2d keypoints to ground truth  
    Inputs:
        - cTr: torch.Tensor (6,). Angle-axis representation of camera to robot pose
        - K: torch.Tensor (3,3). Camera intrinsics
        - kp3d: torch.Tensor (N, 3). 3D points
        - kp2d_gt: torch.Tensor (N, 7). Ground truth and estimated 2D keypoints
        - img: torch.Tensor (3,H,W). Image
        - seg: torch.Tensor (1,H,W). Segmentation
    '''
    cTr_se3 = Se3(Quaternion.from_axis_angle(cTr[...,:3]), cTr[...,3:])
    # kp2d_proj = project_3d_2d(kp3d, K, pose_to_camera=cTr_se3)
    bbox_verts = project_3d_2d(get_bbox(kp3d), K, pose_to_camera=cTr_se3)
    
    img = img.permute(1,2,0).detach().cpu().numpy()[:,:,::-1]
    seg = seg.detach().cpu().numpy()
    plt.figure()
    plt.imshow(denormalize_img(img))
    show_box(bbox_verts, plt.gca())
    if kp2d_gt is not None: show_points(kp2d_gt, plt.gca(), color='red', label='gt', marker_size=130)
    show_points(kp2d_est, plt.gca(), color='white', label='est')
    show_mask(seg, plt.gca())
    plt.legend(); plt.title(title)
    
    if fname is not None: plt.savefig(fname)
    buffer = io.BytesIO(); plt.savefig(buffer, format='png'); buffer.seek(0)
    image = Image.open(buffer).convert('RGB')
    image_arr = np.array(image)
    plt.clf(); plt.close()
    return image_arr
    
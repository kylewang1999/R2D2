'''
File Created:  2023-Sep-18th Mon 5:27
Author:        Kaiyuan Wang (k5wang@ucsd.edu)
Affiliation:   ARCLab @ UCSD
Description:   Test Segment Anything Model on R2D2 Dataset
'''
import sys, os, io, argparse, cv2, torch, yaml, numpy as np
from os.path import join, expanduser, exists
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.utils import make_grid
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from types import SimpleNamespace
from utils import project_3d_2d

from ctrnet.imageloaders.r2d2_data import R2D2DatasetBlock, R2D2DatasetBlockWithSeg, \
    get_camera_intrinsics, get_robot_mesh_files, get_r2d2_dataset, get_kp
from torch.utils.data import DataLoader, ConcatDataset
from config.load import args_from_yaml

BASE_DIR = os.getcwd()
sys.path.append(BASE_DIR)
sys.path.append("..")
from ctrnet.models.CtRNet import CtRNet
from ctrnet.models.BPnP import BPnP
from ctrnet.utils import *
from ctrnet.imageloaders.r2d2_data import R2D2DatasetBlock, \
    get_camera_intrinsics, get_robot_mesh_files, get_r2d2_dataset, get_kp
from segment_anything import sam_model_registry, SamPredictor
# ctrnet.models.CtRNet imports SamPredictor so we reuse that

def get_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")

    args.base_dir = BASE_DIR
    # args.data_folder = expanduser('~/Desktop/data/r2d2_household/pen_out_several')
    args.data_folder = expanduser('~/Desktop/data/r2d2_household/pen_in_several/Fri_Apr_21_10_42_22_2023')
    args.keypoint_seg_model_path = join(args.base_dir,"ctrnet/weights/panda/panda-3cam_azure/net.pth")
    args.urdf_file = join(args.base_dir,"ctrnet/urdfs/Panda/panda.urdf")
    args.sam_checkpoint = expanduser("~/Desktop/models/sam_segment_anything/sam_vit_h_4b8939.pth")
    args.sam_model_type = "vit_h"
    args.device = 'cuda'
    args.use_gpu = True
    args.trained_on_multi_gpus = True

    ##### training parameters #####
    args.batch_size = 1
    args.num_workers = 1
    args.lr = 1e-6
    args.beta1 = 0.9
    args.n_epoch = 500
    args.out_dir = 'results/debug'
    args.ckp_per_epoch = 10
    args.reproj_err_scale = 1.0 / 100.0
    ################################

    # args.robot_name = 'Baxter_left_arm' # "Panda" or "Baxter_left_arm"
    args.robot_name = 'Panda' # "Panda" or "Baxter_left_arm"
    args.n_kp = 7
    args.scale = 0.5
    args.height = 720
    args.width = 1280

    args.camera_ids = ['23404442_left', '23404442_right', '29838012_left','29838012_right']
    args.camera_id = '23404442_right'
    K = get_camera_intrinsics(args.camera_id)
    args.fx, args.fy, args.px, args.py = K[0,0], K[1,1], K[0,2], K[1,2]

    # scale the camera parameters
    args.width = int(args.width * args.scale)
    args.height = int(args.height * args.scale)
    args.fx = args.fx * args.scale
    args.fy = args.fy * args.scale
    args.px = args.px * args.scale
    args.py = args.py * args.scale

    args.trans_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return args

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=2))    

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.8])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def to_device(*args, device='cuda' if torch.cuda.is_available() else 'cpu'):
    return [x.to(device) for x in args]

def to_numpy(*args):
    return [x.detach().cpu().numpy() for x in args]

def kps_to_bbox(kps):
    ''' Get bounding box from keypoints 
    Input:
        - kps: Tensor or Np.array of shape (N,2)
    Output:
        - bbox: Tensor or Np.array of shape (4,), containing [x_min, y_min, x_max, y_max]
    '''
    if isinstance(kps, torch.Tensor): 
        x_min, _ = torch.min(kps[:,:,0], dim=1)
        y_min, _ = torch.min(kps[:,:,1], dim=1)
        x_max, _ = torch.max(kps[:,:,0], dim=1)
        y_max, _ = torch.max(kps[:,:,1], dim=1)
        return torch.stack((x_min, y_min, x_max, y_max), dim=1)
    else:
        return np.array([np.min(kps[..., 0]), np.min(kps[..., 1]), 
                         np.max(kps[..., 0]), np.max(kps[..., 1])])

def project_3d_2d_cv2(points_3d, K, pose_to_camera:(np.array, np.array)):
    '''
    Inputs
        - pose_to_camera: Tuple (rvec, tvec). rvec: (3,), tvec: (3,). 
            Note: rvec encodes rotation as axis-angle.
    '''
    
    imagePoints, _ = cv2.projectPoints(
        points_3d, 
        pose_to_camera[0], 
        pose_to_camera[1], K, None)

    return imagePoints



def try_sam():
    ''' Get Data '''
    args = get_args()
    model = CtRNet(args)
    mesh_files = get_robot_mesh_files()[0:1]    # Only keep base
    # mesh_files = get_robot_mesh_files()

    robot_renderer = model.setup_robot_renderer(mesh_files)

    # dataset = get_r2d2_dataset(args.data_folder, camera_ids= args.camera_ids, max_num_blocks=1,
    #                            trans_to_tensor=trans_to_tensor)
    dataset = R2D2DatasetBlock(
        data_folder=args.data_folder, 
        camera_id=args.camera_id, 
        trans_to_tensor=args.trans_to_tensor,
        n_kp=args.n_kp,
    )

    # Hand-annotated 3d keypoint locations in the robot frame. Shape (N,2)
    # kp3d = torch.from_numpy(get_kp(camera_id=args.camera_id, type='3d'))\
        # .to(device=args.device, dtype=torch.float32)
    kp3d = get_kp(camera_id=args.camera_id, type='3d')

    kp2d_gt = dataset.kp2d_gt
    inval = np.isnan(kp2d_gt).any(axis=1).to(dtype=torch.bool)
    kp2d_gt = kp2d_gt[~inval]
    kp3d = kp3d[~inval]
    bbox = kps_to_bbox(kp2d_gt).to(args.device)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    img, joint_angle, kp2d_gt = next(iter(dataloader))   # (B,3,H=720,W=1280), (B,7)
    img, joint_angle, kp2d_gt = to_device(img[0], joint_angle[0], kp2d_gt[0])
    joint_angle = joint_angle[0:1]  # Only keep base

    # for img, joint_angle in tqdm(dataloader):
    #     pass

    ''' Show Points '''
    input_points = kp2d_gt.detach().cpu().numpy()
    input_labels = np.ones_like(input_points[:, 0])
    plt.figure(figsize=(10,10))
    plt.imshow(img.cpu().numpy().transpose(1,2,0)[...,::-1])  # (BGR to RGB)
    show_points(input_points, input_labels, plt.gca())
    plt.axis('on')
    plt.show(); plt.savefig("foo.png"); plt.clf()

    # ''' Predict SAM Segmentation '''
    img = (img*255).to(dtype=torch.uint8)
    img = img.cpu().numpy().transpose(1,2,0)[...,::-1]  # (BGR to RGB)

    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam.sam_checkpoint)
    # sam.to(device=args.device)
    sam.to(device='cpu')
    predictor = SamPredictor(sam)
    bbox = bbox.detach().cpu().numpy()
    predictor.set_image(img)
    mask, _, _ = predictor.predict(
        # point_coords=torch.from_numpy(input_points).to(device=args.device)[None,...], 
        # point_labels=torch.from_numpy(input_labels).to(device=args.device)[None,...],
        # point_coords= input_points,
        # point_labels=input_labels,
        box=bbox[None,...],
        multimask_output=False,
    )

    # cv2.imwrite('mask.png', mask.squeeze().astype(np.uint8)*255)

    plt.figure(figsize=(10,10))
    plt.imshow(img)
    show_mask(mask, plt.gca(), random_color=False) 
    show_points(input_points, input_labels, plt.gca())
    show_box(bbox, plt.gca())
    plt.axis('off'); plt.savefig('mask2.png'); plt.clf()


    ''' PnP Solver '''
    # Camera intrinsic matrix of hspae (3,3)
    K = get_camera_intrinsics(camera_id=args.camera_id)

    pose_ctr = BPnP.apply(
        torch.from_numpy(kp2d_gt[None,...]).to(device=args.device, dtype=torch.float32), 
        torch.from_numpy(kp3d).to(device=args.device, dtype=torch.float32), 
        torch.from_numpy(K).to(device=args.device, dtype=torch.float32))


    kp2d_est = project_3d_2d_cv2(kp3d, K, 
        (pose_ctr.squeeze()[:3].detach().cpu().numpy(), 
         pose_ctr.squeeze()[3:].detach().cpu().numpy()))
    kp2d_est = np.squeeze(kp2d_est)
    
    img_np = img / 255.0
    img_np = 0.0* np.ones(img_np.shape) + img_np * 0.6
    img_np = overwrite_image(img_np,kp2d_est.astype(int), color=(0,1,0))

    robot_mesh = robot_renderer.get_robot_mesh(joint_angle.detach().cpu().numpy())
    rendered_image = model.render_single_robot_mask(pose_ctr.squeeze(), robot_mesh, robot_renderer)

    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.title("keypoints")
    plt.imshow(img_np)
    plt.subplot(1,2,2)
    plt.title("rendering")
    plt.imshow(rendered_image.squeeze().detach().cpu().numpy())
    plt.show(); plt.savefig('kp_render.png')

def dict_to_object(d):
    if isinstance(d, dict): return SimpleNamespace(**{k: dict_to_object(v) for k, v in d.items()})
    elif isinstance(d, list): return [dict_to_object(x) for x in d]
    else: return d

def replace_fstring_in_dict(d):
    for key, value in d.items():
        if isinstance(value, dict):
            replace_fstring_in_dict(value)
        elif isinstance(value, str):
            d[key] = value.format_map(d)


def batch_items_to_sam_format(images, kp2ds, bboxes):
    ''' Convert a batch of img and bbox from dataloader to SAM format. 
    Inputs:
        - img: Tensor (B,3,H,W)
        - bbox: Tensor (B,4)
    Reurns: A list (of length B) of dicts.
    '''
    sam_input_batch = [
        {'image': img,
         'boxes': bbox[None,...],
        #  'point_coords': kp2d[None,...],
        #  'point_labels': torch.ones_like(kp2d[:,0])[None,...],
         'original_size': img.shape[1:]
        } for img, kp2d, bbox in zip(images, kp2ds, bboxes)
    ]
    return sam_input_batch

def infer_masks_base_only(predictor:SamPredictor, args, visualize=False):

    for camera_id in args.r2d2.camera_ids:
        args.r2d2.camera_id = camera_id
        target_fname = join(args.r2d2.data_folder, f'{args.r2d2.camera_id}_seg.npz')

        # 1. Get Dataset and Dataloader
        dataset = R2D2DatasetBlock(
            data_folder =       args.r2d2.data_folder, 
            camera_id =         args.r2d2.camera_id, 
            trans_to_tensor =   args.r2d2.trans_to_tensor,
            n_kp =              args.robot.n_kp
        )
        dataloader = DataLoader(dataset, 
            batch_size =    args.r2d2.batch_size, 
            num_workers =   args.r2d2.num_workers,
            shuffle =       args.r2d2.shuffle
        )

        # 2. Load img, joint_angle, kp2d_gt, bbox: (B,3,H,W), (B,7), (B,7,2), (B,4)
        mask_list = []
        print(f'Infering... Target file: {target_fname:^10}')
        for img_batch, joint_angle_batch, kp2d_gt_batch in tqdm(dataloader): 
            assert len(img_batch) == 1, f'Only batch size 1 is supported. {len(img_batch)} images in a batch is not supported'
            inval = np.isnan(kp2d_gt_batch).any(axis=-1).to(dtype=torch.bool)
            kp2d_gt_batch = kp2d_gt_batch[~inval,:].reshape([len(img_batch),-1,2])
            bbox_batch = kps_to_bbox(kp2d_gt_batch)
            img_batch, joint_angle_batch, kp2d_gt_batch, bbox_batch = to_device(
                img_batch,            # (B,3,H,W)
                joint_angle_batch,    # (B,7)
                kp2d_gt_batch,        # (B,7-1,2)  # 1 invalid keypoint, 6 keypoints
                bbox_batch,           # (B,4)      # For each image: (xmin, ymin, xmax, ymax)
                device=args.exp.device
            )

            img, bbox = to_numpy(img_batch.squeeze(), bbox_batch.squeeze())
            img = (img.transpose(1,2,0)[...,::-1] * 255).astype(np.uint8)
            predictor.set_image(img)
            mask, _, _ = predictor.predict(
                box=bbox[None,...],
                multimask_output=False,
            )
            mask_list.append(mask.squeeze())

            if visualize:
                plt.figure(figsize=(10,10))
                plt.imshow(img)
                show_mask(mask, plt.gca(), random_color=False) 
                show_box(bbox, plt.gca())
                plt.savefig('foo.png'); plt.clf(); plt.close()

            np.savez_compressed(file=target_fname, masks=np.stack(mask_list))

''' Propmt for segmenting 2 joints '''
# PROMPT_TABLE = {
#     '23404442_left':{'bbox': np.array([0,100,400,500]), 'points':np.array([[300,450]]), 'labels':np.array([1])},
#     '23404442_right':{'bbox':np.array([0,100,300,500]), 'points':np.array([[200,400]]), 'labels':np.array([1])}, 
#     '29838012_left':{'bbox':np.array([950,300,1280,600]), 'points':np.array([[1000,500], [1150,350]]), 'labels':np.array([1,1])},
#     '29838012_right':{'bbox':np.array([800,300,1280,600]), 'points':np.array([[900,500],[1000,350]]), 'labels':np.array([1,1])}
# }

# ''' Propmt for segmenting 3 joints '''
# PROMPT_TABLE = {
#     '23404442_left':{'bbox': np.array([0,100,400,500]), 'points':np.array([[300,450],[200,200]]), 'labels':np.array([1,1])},
#     '23404442_right':{'bbox':np.array([0,100,300,500]), 'points':np.array([[200,400],[100,200]]), 'labels':np.array([1,1])}, 
#     '29838012_left':{'bbox':np.array([950,300,1280,600]), 'points':np.array([[1000,500], [1150,350]]), 'labels':np.array([1,1])},
#     '29838012_right':{'bbox':np.array([800,300,1280,600]), 'points':np.array([[900,500],[1000,350]]), 'labels':np.array([1,1])}
# }

''' cTr pose for Fri_Apr_21_10_42_22_2023 estimated by properly trained cTrNet '''
CTR_POSE_TABLE = {
    '23404442_left' : np.array([1.8219, -0.2591, 0.1965, -0.4622, 0.1534, 0.7621]), 
    '23404442_right' : np.array([1.7727, -0.4724, 0.3734, -0.6855, 0.1627, 0.8416]), 
    '29838012_left' : np.array([0.6690, -2.1279, 1.8204,  0.5540, 0.3119, 0.8275]),
    '29838012_right' : np.array([0.6671, -2.2482, 1.7823,  0.4002, 0.2975, 0.8104]) 
}


def infer_masks_all_joints(predictor:SamPredictor, args, visualize=False):

    ''' Prepare args '''
    K = get_camera_intrinsics(args.r2d2.camera_id)
    args.ctrnet.fx  = K[0,0]*args.r2d2.scale; args.ctrnet.fy = K[1,1]*args.r2d2.scale
    args.ctrnet.px = K[0,2]*args.r2d2.scale; args.ctrnet.py = K[1,2]*args.r2d2.scale
    args.ctrnet.K = np.array([
        [args.ctrnet.fx, 0, args.ctrnet.px], [0, args.ctrnet.fy, args.ctrnet.py], [0,0,1]
    ])
    args.ctrnet.width = args.r2d2.width; 
    args.ctrnet.height = args.r2d2.height
    args.ctrnet.device = args.exp.device
    args.ctrnet.log_dir = args.exp.log_dir
    kp_annotation_dict = yaml.safe_load(open(args.r2d2.kp_annotation_yaml, 'r'))
    writer = SummaryWriter(join(args.exp.log_dir, 'tensorboard'))

    from ctrnet.models.robot_arm import PandaArm
    model = CtRNet(args.ctrnet)
    panda_arm = PandaArm(args.ctrnet.urdf_file)
    robot_renderer = model.setup_robot_renderer(get_robot_mesh_files()[:3])

    # infer_masks_base_only(predictor, args)
    data_base_dir = "/".join(args.r2d2.data_folder.split('/')[:-1])
    iter_count = 0
    for data_folder in args.r2d2.annotated_subfolders[::-1]:
        print(f'==== Infering masks for {data_folder} ====')
        args.r2d2.data_folder = join(data_base_dir, data_folder)
    
        for camera_id in args.r2d2.camera_ids:

            annotation_id = f'{args.r2d2.data_folder.split("/")[-1]}-{camera_id.split("_")[0]}-{camera_id.split("_")[1]}'
            if not annotation_id in kp_annotation_dict['valid_trails']:
                print(f'{annotation_id} not in valid trails. Skipping...')
                continue
            annotation_info = kp_annotation_dict[annotation_id]

            target_fname = join(args.r2d2.data_folder, f'{camera_id}_seg_3joints.npz')

            dataset = R2D2DatasetBlock(
                data_folder =       args.r2d2.data_folder, 
                camera_id =         camera_id, 
                trans_to_tensor =   args.r2d2.trans_to_tensor,
                n_kp =              args.ctrnet.n_kp,
                scale =             args.r2d2.scale,
                kp2ds_info_dict = annotation_info
            ) 
            dataloader = DataLoader(dataset, 
                batch_size =    args.r2d2.batch_size, 
                num_workers =   args.r2d2.num_workers,
                shuffle =       args.r2d2.shuffle
            )
            
            mask_list = []
            for batch_id, (img_batch, joint_angle_batch, extrinsic_batch, kp2d_gt_batch) in enumerate(tqdm(dataloader)): 
                assert len(img_batch) == 1, f'Only batch size 1 is supported. {len(img_batch)} images in a batch is not supported'
                img = img_batch.squeeze().detach().cpu().numpy()
                img = (img.transpose(1,2,0)[...,::-1] * 255).astype(np.uint8)
                predictor.set_image(img)
                
                ctr_pose = torch.tensor(CTR_POSE_TABLE[camera_id], dtype=torch.float32, device=args.exp.device)
                robot_mesh = robot_renderer.get_robot_mesh(joint_angle_batch[0].detach().cpu().numpy())
                img_rendered = model.render_single_robot_mask(ctr_pose, robot_mesh, robot_renderer) # (1,H,W)

                ''' Generate SAM bbox prompt based on rendered robot mesh (works poorly)'''
                # rows, cols = torch.where(img_rendered.squeeze() == 1)
                # bbox = torch.tensor([cols.min(), rows.min(), cols.max(), rows.max()]).detach().cpu().numpy()
                
                ''' Use pose initialization from cTrNet and projected 2d joint keypoints (works poorly)'''
                # from kornia.geometry.liegroup import Se3
                # from kornia.geometry.quaternion import Quaternion
                # _, joint_locs_3d = panda_arm.get_joint_RT(joint_angle_batch.squeeze().detach().cpu().numpy())
                # joint_locs_2d = project_3d_2d(
                #     torch.tensor(joint_locs_3d, dtype=torch.float32, device=args.exp.device),
                #     torch.tensor(args.ctrnet.K, dtype=torch.float32, device=args.exp.device), 
                #     Se3(Quaternion.from_axis_angle(ctr_pose[...,:3]), ctr_pose[...,3:])
                # )
                # bbox = torch.tensor(np.array([[0,0,400,500], [500,0,800,200]])).to(args.exp.device)
                # points = joint_locs_2d[None,...]
                # labels = torch.ones(len(joint_locs_2d), dtype=torch.uint8)[None,...]
                # mask, _, _ = predictor.predict_torch(
                #     point_coords=points,
                #     point_labels=labels,
                #     # boxes=bbox,
                #     multimask_output=False,
                # )

                ''' Use visually inspected points and bbox '''
                points = np.array(annotation_info['kp_sam_prompt'])
                labels = np.ones(len(points), dtype=np.uint8)
                bbox = np.array(annotation_info['bbox'])
                mask, _, _ = predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    box=bbox,
                    multimask_output=False,
                )
                mask = mask.squeeze()
                mask_list.append(mask)

                if isinstance(mask, torch.Tensor): mask = mask.squeeze().detach().cpu().numpy()
                if isinstance(bbox, torch.Tensor): bbox = bbox.squeeze().detach().cpu().numpy()
                if isinstance(points, torch.Tensor): points = points.squeeze().detach().cpu().numpy()
                if isinstance(labels, torch.Tensor): labels = labels.squeeze().detach().cpu().numpy()
                
                if visualize and batch_id % 16 == 0:
                    plt.figure(figsize=(10,10))
                    plt.imshow(img); plt.grid(True)
                    show_mask(mask, plt.gca(), random_color=False) 
                    show_box(bbox, plt.gca())
                    plt.title(f'{data_folder}-{camera_id}')
                    show_points(points, labels, plt.gca(), marker_size=130)
                    plt.savefig(f'{data_folder}-{camera_id}.png')
                    buffer = io.BytesIO(); plt.savefig(buffer, format='png'); buffer.seek(0); plt.clf(); plt.close()
                    image_arr = np.array(Image.open(buffer).convert('RGB'))
                    writer.add_image("eval_img/masknpose_pred", torch.cat([torch.tensor(image_arr).permute(2,0,1)] ,dim=-1), global_step=iter_count)    
                    writer.add_image("eval_img/render", torch.cat([img_rendered.repeat(3,1,1)], dim=-1), global_step= iter_count)
                iter_count += 1
            print(f'==== Writing to {target_fname} ====')
            np.savez_compressed(file=target_fname, masks=np.stack(mask_list))

if __name__ == '__main__':
    # TODO: Infer mask for all keypoint-annotated r2d2 subfolders    
    # TODO: Infer mask for entire kinematic chain
    base_yaml = './config/base_sam_maskgen.yaml'
    args = args_from_yaml(base_yaml)
    
    K = get_camera_intrinsics(args.r2d2.camera_id)
    args.ctrnet.fx  = K[0,0]*args.r2d2.scale; args.ctrnet.fy = K[1,1]*args.r2d2.scale
    args.ctrnet.px = K[0,2]*args.r2d2.scale; args.ctrnet.py = K[1,2]*args.r2d2.scale
    args.ctrnet.K = np.array([
        [args.ctrnet.fx, 0, args.ctrnet.px], [0, args.ctrnet.fy, args.ctrnet.py], [0,0,1]
    ])
            

    sam = sam_model_registry[args.sam.model_type](checkpoint=args.sam.checkpoint)
    sam.to(device=args.exp.device)
    predictor = SamPredictor(sam)

    infer_masks_all_joints(predictor, args, visualize=True)
    

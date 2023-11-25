import os, torch, numpy as np
from os.path import exists, join
from tqdm import tqdm

from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ctrnet.models.CtRNet import CtRNet, CtRNetBaseOnly
from ctrnet.imageloaders.r2d2_data import R2D2DatasetBlock, R2D2DatasetBlockWithSeg, \
    get_camera_intrinsics, get_robot_mesh_files, get_r2d2_dataset, get_kp
from config.load import args_from_yaml
from utils import *

''' cTr pose for Fri_Apr_21_10_42_22_2023 estimated by properly trained cTrNet '''
CTR_POSE_TABLE = {
    '23404442_left' : [ 1.8219, -0.2591,  0.1965, -0.4622,  0.1534,  0.7621], 
    '23404442_right' : [ 1.7727, -0.4724,  0.3734, -0.6855,  0.1627,  0.8416], 
    '29838012_left' : [ 0.6690, -2.1279,  1.8204,  0.5540,  0.3119,  0.8275],
    '29838012_right' : [ 0.6671, -2.2482,  1.7823,  0.4002,  0.2975,  0.8104] 
}

def train_model_pretrain_backbones(args, model, dataloader):
    model.train()

    tblog_dir = join(args.log_dir, 'tensorboard')
    print(f'\n ==== Tensorboard log dir: {tblog_dir:^5} ====')
    writer = SummaryWriter(tblog_dir)

    keypoint_seg_predictor = model.keypoint_seg_predictor

    criterionMSE_sum = torch.nn.MSELoss(reduction='sum')
    criterionMSE_mean = torch.nn.MSELoss(reduction='mean')
    criterionBCE = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(keypoint_seg_predictor.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

    kp3d = torch.from_numpy(get_kp(type='3d')).to(model.device, dtype=torch.float32)
    K = torch.from_numpy(args.K).to(device=model.device, dtype=torch.float32)

    def train_one_epoch(writer, time_step=0):

        loss_kp_averager = AverageMeter()
        loss_seg_averager = AverageMeter()
        loss_averager = AverageMeter()

        for batch_id, (img, joint_angle, kp2d_gt, extrinsic, mask_gt) in enumerate(pbar:=tqdm(dataloader)):    
            img, joint_angle, kp2d_gt, mask_gt = to_device(
                img, joint_angle, kp2d_gt, mask_gt, device=args.device)

            img = img.to(args.device)
            joint_angle = joint_angle.to(args.device)
            kp2d_gt = kp2d_gt.to(args.device)

            kp2d, seg = keypoint_seg_predictor(img)     # Where is BPNP involved? Visualize rendered image?
            
            valid = torch.logical_and(torch.logical_and(kp2d_gt[:,:,0] < args.width, kp2d_gt[:,:,0] > 0), 
                torch.logical_and(kp2d_gt[:,:,1] < args.height, kp2d_gt[:,:,1] > 0))

            loss_seg = criterionBCE(seg.squeeze(), mask_gt.to(dtype=seg.dtype))
            loss_kp = criterionMSE_mean(kp2d_gt[valid], kp2d[valid])
            loss = args.kp_loss_weight * loss_kp + loss_seg

            optimizer.zero_grad()
            loss.backward(); optimizer.step(); scheduler.step(loss)
            torch.nn.utils.clip_grad_value_(keypoint_seg_predictor.parameters(), 10)

            loss_kp_avg = loss_kp_averager.update(loss_kp.item())
            loss_seg_avg = loss_seg_averager.update(loss_seg.item())
            loss_avg = loss_averager.update(loss.item())
            pbar.set_postfix({'epoch':time_step, 'batch_id':batch_id, 'loss':loss_avg})
        
            if batch_id % 8 == 0:
                with torch.no_grad():
                    ''' Visualize 1 image from the last batch'''
                    cTr, kp2d, seg = model.inference_batch_images_base_only(img, kp3d.repeat(img.shape[0], 1, 1))
                    cTr_gtpoints = model.bpnp_m3d(kp2d_gt[:,:-1,:], kp3d.repeat(img.shape[0], 1, 1)[:,:-1,:], K)
                    pose_and_seg_pred = plot_pose_and_gtkp(cTr[0], K, kp3d, kp2d[0], kp2d_gt[0], img[0], seg[0], fname=None, title='Predicted pose and segmentation')
                    pose_and_seg_gt = plot_pose_and_gtkp(cTr_gtpoints[0], K, kp3d, kp2d[0], kp2d_gt[0], img[0], mask_gt[0], fname=None, title='GT pose and segmentation')
                    writer.add_image("train_img/masknpose_pred|gt", torch.cat([torch.tensor(pose_and_seg_pred).permute(2,0,1), torch.tensor(pose_and_seg_gt).permute(2,0,1)] ,dim=-1), global_step=time_step*dataloader.batch_size+batch_id+batch_id)
                    writer.add_image("train_img/seg_pred|gt|img", torch.cat([seg[0].repeat(3,1,1), mask_gt[0][None,...].to(dtype=seg.dtype).repeat(3,1,1), img[0][[2,1,0],:,:]], dim=-1), global_step=time_step*dataloader.batch_size+batch_id+batch_id)
                    writer.add_scalar('train_scalar/loss', loss_avg, global_step=time_step*dataloader.batch_size+batch_id+batch_id)
                    writer.add_scalar('train_scalar/loss_kp', loss_kp_avg, global_step=time_step*dataloader.batch_size+batch_id+batch_id)

    try:    
        print(f'\nStart training... Total epochs: {args.n_epoch}')
        for epoch in range(args.n_epoch): 
            train_one_epoch(writer, epoch) 
            if epoch > 0 and epoch % args.ckp_per_epoch == 0: 
                checkpoint_path = '/'.join(args.checkpoint_path.split('/')[:-1]) + f'/ckp_{epoch}.pth'
                print(f'\t At epcoh {epoch}. Saving model checkpoint to {checkpoint_path}')
                torch.save(model.state_dict(), checkpoint_path)
        torch.save(model.state_dict(), args.checkpoint_path)
    except KeyboardInterrupt: 
        checkpoint_path = '/'.join(args.checkpoint_path.split('/')[:-1]) + f'/interrupted_{epoch}.pth'
        print(f'Training interrupted by user. Completed {epoch} / {args.n_epoch} epochs.')
        print(f'Saving model to {checkpoint_path}')
        torch.save(model.state_dict(), checkpoint_path)

def train_model_self_supervised(args, model:CtRNet, dataloader):
    model.keypoint_seg_predictor.train()

    # 1. Robot mesh, renderer, 3d keypoints, camera intrinsic, SummaryWriter
    robot_renderer = model.setup_robot_renderer(get_robot_mesh_files()[:3])
    kp3d = torch.Tensor(get_kp(type='3d')).to(model.device, dtype=torch.float32)
    kp3d = kp3d.repeat(dataloader.batch_size, 1, 1)
    K = torch.Tensor(args.K).to(device=model.device, dtype=torch.float32)

    writer = SummaryWriter(join(args.log_dir, 'tensorboard'))
    print(f'\n ==== Tensorboard log dir: {writer.get_logdir():^5} ====')

    # 2. Criterion and optimizer
    criterionMSE_sum = torch.nn.MSELoss(reduction='sum')
    criterionMSE_mean = torch.nn.MSELoss(reduction='mean')
    criterionBCE = torch.nn.BCEWithLogitsLoss()
    criterions = {"mse_sum": criterionMSE_sum, "mse_mean": criterionMSE_mean, "bce": criterionBCE}

    optimizer = optim.Adam(model.keypoint_seg_predictor.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

    def train_one_epoch(writer, time_step=0):
        
        loss_averager = AverageMeter()
        loss_mse_averager = AverageMeter()
        loss_bce_averager = AverageMeter()

        for batch_id, (img, joint_angle, kp2d_gt, extrinsic, mask_gt) in enumerate(pbar:=tqdm(dataloader)):    
            img, joint_angle, kp2d_gt, mask_gt = to_device(
                img, joint_angle, kp2d_gt, mask_gt, device=args.device)
            
            loss, loss_mse, loss_bce, rendered_mask  = model.train_on_batch(
                img, joint_angle.cpu().squeeze(), robot_renderer, kp3d[0], criterions, 'train')

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_value_(model.keypoint_seg_predictor.parameters(), 10)
            optimizer.step(); scheduler.step(loss)

            loss_avg = loss_averager.update(loss.item())
            loss_mse_avg = loss_mse_averager.update(loss_mse.item())
            loss_bce_avg = loss_bce_averager.update(loss_bce.item())
            pbar.set_postfix({'epoch':time_step, 'batch_id':batch_id, 'loss':loss_avg})

            if batch_id % 8 == 0:
                ''' Visualize 1 image from the last batch'''
                with torch.no_grad():
                    cTr, kp2d, seg = model.inference_batch_images_base_only(img, kp3d)
                    cTr_gtpoints = model.bpnp_m3d(kp2d_gt[:,:-1,:], kp3d[:,:-1,:], K)
                    pose_and_seg_pred = plot_pose_and_gtkp(cTr[0], K, kp3d[0], kp2d[0], kp2d_gt[0], img[0], seg[0], fname=None, title='Predicted pose and segmentation')
                    pose_and_seg_gt = plot_pose_and_gtkp(cTr_gtpoints[0], K, kp3d[0], kp2d[0], kp2d_gt[0], img[0], mask_gt[0], fname=None, title='GT pose and segmentation')
                    writer.add_image("train_img/masknpose_pred|gt", torch.cat([torch.tensor(pose_and_seg_pred).permute(2,0,1), torch.tensor(pose_and_seg_gt).permute(2,0,1)] ,dim=-1), global_step=time_step*dataloader.batch_size+batch_id)
                    writer.add_image("train_img/seg_|pred|gt|render|img|", torch.cat([
                        seg[0].repeat(3,1,1), 
                        mask_gt[0][None,...].to(dtype=seg.dtype).repeat(3,1,1), 
                        rendered_mask[0][None,...].to(dtype=seg.dtype).repeat(3,1,1),
                        img[0][[2,1,0],:,:]
                    ], dim=-1), global_step=time_step*dataloader.batch_size+batch_id)
                    writer.add_scalar('train_scalar/loss', loss_avg, global_step=time_step*dataloader.batch_size+batch_id)
                    writer.add_scalar('train_scalar/loss_mse', loss_mse_avg, global_step=time_step*dataloader.batch_size+batch_id)
                    writer.add_scalar('train_scalar/loss_bce', loss_bce_avg, global_step=time_step*dataloader.batch_size+batch_id)

    try:    
        print(f'\nStart training... Total epochs: {args.n_epoch}')
        for epoch in range(args.n_epoch): 
            train_one_epoch(writer, epoch) 
            if epoch > 0 and epoch % args.ckp_per_epoch == 0: 
                checkpoint_path = '/'.join(args.checkpoint_path.split('/')[:-1]) + f'/ckp_{epoch}.pth'
                print(f'\t At epcoh {epoch}. Saving model checkpoint to {checkpoint_path}')
                torch.save(model.state_dict(), checkpoint_path)
        torch.save(model.state_dict(), args.checkpoint_path)
    except KeyboardInterrupt: 
        checkpoint_path = '/'.join(args.checkpoint_path.split('/')[:-1]) + f'/interrupted_{epoch}.pth'
        print(f'Training interrupted by user. Completed {epoch} / {args.n_epoch} epochs.')
        print(f'Saving model to {checkpoint_path}')
        torch.save(model.state_dict(), checkpoint_path)

def eval_model(args, model, dataloader):
    model.eval()
    writer = SummaryWriter(join(args.log_dir, 'tensorboard'))
    
    kp3d = torch.from_numpy(get_kp(type='3d')).to(model.device, dtype=torch.float32)
    kp3d = kp3d.repeat(dataloader.batch_size, 1, 1)
    K = torch.from_numpy(args.K).to(device=model.device, dtype=torch.float32)

    batch_id = 0
    with torch.no_grad():
        for batch_id, (img, joint_angle, extrinsic, kp2d_gt) in enumerate(pbar:=tqdm(dataloader)):
            img, joint_angle, kp2d_gt, seg_gt = to_device(img, joint_angle, kp2d_gt, seg_gt)
            cTr, kp2d, seg = model.inference_batch_images_base_only(img, kp3d)
            cTr_gtpoints = model.bpnp_m3d(kp2d_gt[:,:-1,:], kp3d[:,:-1,:], K)

            if batch_id % 4 == 0:
                pose_and_seg_pred = plot_pose_and_gtkp(cTr[0], K, kp3d[0], kp2d[0], kp2d_gt[0], img[0], seg[0], fname=None, title='Predicted pose and segmentation')
                pose_and_seg_gt = plot_pose_and_gtkp(cTr_gtpoints[0], K, kp3d[0], kp2d[0], kp2d_gt[0], img[0], seg_gt[0], fname=None, title='GT pose and segmentation')
                writer.add_image("eval_pose_and_seg_pred", torch.tensor(pose_and_seg_pred).permute(2,0,1), global_step=batch_id)
                writer.add_image("eval_img/masknpose_pred|gt", torch.cat([torch.tensor(pose_and_seg_pred).permute(2,0,1), torch.tensor(pose_and_seg_gt).permute(2,0,1)] ,dim=-1), global_step=batch_id)
                writer.add_image("eval_pose_and_seg_gt", torch.tensor(pose_and_seg_gt).permute(2,0,1), global_step=batch_id)
                writer.add_image("eval_img/seg_pred|gt|img", torch.cat([
                    seg[0].repeat(3,1,1), 
                    seg_gt[0][None,...].to(dtype=seg.dtype).repeat(3,1,1), 
                    img[0][[2,1,0],:,:]
                ], dim=-1), global_step=batch_id)

def eval_model_nogt(args, model, dataloader):
    model.eval()
    robot_renderer = model.setup_robot_renderer(get_robot_mesh_files()[:3])
    writer = SummaryWriter(join(args.log_dir, 'tensorboard'))
    
    kp3d = torch.from_numpy(get_kp(type='3d')).to(model.device, dtype=torch.float32)
    kp3d = kp3d.repeat(dataloader.batch_size, 1, 1)
    K = torch.from_numpy(args.K).to(device=model.device, dtype=torch.float32)

    batch_id = 0
    with torch.no_grad():
        for batch_id, (img, joint_angle, extrinsic) in enumerate(pbar:=tqdm(dataloader)):
            img, joint_angle, extrinsic = to_device(img, joint_angle, extrinsic)
            cTr, kp2d, seg = model.inference_batch_images_base_only(img, kp3d)
            
            if batch_id % 4 == 0:
                ''' r2d2 extrinsic information '''
                # from transforms3d.euler import euler2mat, mat2euler
                # cTr_gt = torch.concat([extrinsic[0,3:], extrinsic[0,:3]])
                # cTr_gt = cTr_gt.detach().cpu().numpy() 
                # R = euler2mat(cTr_gt[0], cTr_gt[1], cTr_gt[2], 'sxyz')
                # t_inv = - cTr_gt[3:]
                # R_inv = np.linalg.inv(R)
                # angles_inv = mat2euler(R_inv, 'sxyz')
                # reverse_extrinsic = np.concatenate([angles_inv, t_inv])
                # reverse_extrinsic = torch.tensor(reverse_extrinsic, dtype=torch.float32)
                # R_pred = euler2mat(*cTr[0][:3].detach().cpu().numpy())
                # pose_and_seg_pred = plot_pose_and_gtkp(cTr_gt, K, kp3d[0], kp2d[0], None, img[0], seg[0], fname=None, title='Predicted pose and segmentation')
                # pose_and_seg_pred = plot_pose_and_gtkp(reverse_extrinsic, K, kp3d[0], kp2d[0], None, img[0], seg[0], fname=None, title='Predicted pose and segmentation')
                
                pose_and_seg_pred = plot_pose_and_gtkp(cTr[0], K, kp3d[0], kp2d[0], None, img[0], seg[0], fname=None, title='Predicted pose and segmentation')
                robot_mesh = robot_renderer.get_robot_mesh(joint_angle[0].detach().cpu().numpy())
                img_rendered = model.render_single_robot_mask(cTr[0], robot_mesh, robot_renderer)
                writer.add_image("eval_img/masknpose_pred", torch.cat([torch.tensor(pose_and_seg_pred).permute(2,0,1)] ,dim=-1), global_step=batch_id)
                writer.add_image("eval_img/seg_pred|render", torch.cat([
                    seg[0].repeat(3,1,1),
                    img_rendered.repeat(3,1,1)
                    # img[0][[2,1,0],:,:]
                ], dim=-1), global_step=batch_id)

# if __name__ == "__main__":
#     from transforms3d.euler import euler2mat, mat2euler
#     from transforms3d.affines import compose
#     pose0 = torch.Tensor([-2.1969, -0.1095, -0.3595,  0.3367, -0.4788,  0.5599], dtype=torch.float32)
#     pose1 = torch.Tensor([ 1.7999, -0.2127,  0.1898, -0.4245,  0.1525,  0.7185], dtype=torch.float32)
#     angle_euler = pose1[:3]
#     translation = pose1[3:]
#     R = euler2mat(angle_euler[0], angle_euler[1], angle_euler[2])


if __name__ == "__main__":

    base_yaml = './config/base.yaml'
    # aux_yaml = './config/ctrnet_pretrain-sam-mask-supervised_all4cams_seg3joints.yaml'    # pre-train
    aux_yaml = './config/ctrnet_train_self-supervised.yaml'  # train (self-supervised)
    # aux_yaml = './config/ctrnet_eval-unseen.yaml'

    args = args_from_yaml(base_yaml, aux_yaml)    # For eval

    if not exists(args.exp.log_dir): os.makedirs(args.exp.log_dir)

    # 0. Prepare args
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

    # 1. Get Model and mesh files
    model = CtRNetBaseOnly(args.ctrnet)
    
    # 2. Get Dataset and Dataloader
    dataset = ConcatDataset([
        R2D2DatasetBlockWithSeg(
            data_folder =       args.r2d2.data_folder, 
            camera_id =         cam_id, 
            trans_to_tensor =   args.r2d2.trans_to_tensor,
            n_kp =              args.ctrnet.n_kp,
            scale =             args.r2d2.scale
        ) for cam_id in args.r2d2.camera_ids
    ]) if 'eval' not in aux_yaml else ConcatDataset([
        R2D2DatasetBlock(
            data_folder =       args.r2d2.data_folder, 
            camera_id =         cam_id, 
            trans_to_tensor =   args.r2d2.trans_to_tensor,
            n_kp =              args.ctrnet.n_kp,
            scale =             args.r2d2.scale
        ) for cam_id in args.r2d2.camera_ids
    ]) 

    dataloader = DataLoader(dataset, 
        batch_size =    args.r2d2.batch_size, 
        num_workers =   args.r2d2.num_workers,
        shuffle =       args.r2d2.shuffle
    )

    # 3. Train or Eval
    if 'pretrain' in aux_yaml:
        train_model_pretrain_backbones(args.ctrnet, model, dataloader)
    elif 'train' in aux_yaml:
        model.load_state_dict(torch.load(args.ctrnet.checkpoint_path))
        print(f'Loaded CtRNet checkpoint from {args.ctrnet.checkpoint_path}')
        train_model_self_supervised(args.ctrnet, model, dataloader)
    else:
        model.load_state_dict(torch.load(args.ctrnet.checkpoint_path))
        print(f'Loaded CtRNet checkpoint from {args.ctrnet.checkpoint_path}')
        eval_model_nogt(args.ctrnet, model, dataloader)
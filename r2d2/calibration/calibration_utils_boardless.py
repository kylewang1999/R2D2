'''
File Created:  2023-Dec-5th Tue 12:54
Author:        Kaiyuan Wang (k5wang@ucsd.edu)
Affiliation:   ARCLab @ UCSD
Description:   Off-line boardless calibration utils.
'''
import os, sys, yaml, torch, numpy as np
import torchvision.transforms as transforms
from os.path import abspath, join, expanduser
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from types import SimpleNamespace

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))  # for resolving r2d2.* package in unit testing
from r2d2.calibration.ctrnet.models.CtRNet import CtRNetBaseOnly
from r2d2.calibration.ctrnet.imageloaders.r2d2_data import R2D2DatasetBlock
from r2d2.calibration.ctrnet.utils import to_device, plot_pose_and_gtkp, average_euler_angles

def dict2object(d):
    if isinstance(d, dict): return SimpleNamespace(**{k: dict2object(v) for k, v in d.items()})
    elif isinstance(d, list): return [dict2object(x) for x in d]
    else: return d

def yaml2namespace(yaml_file):
    args_dict = yaml.safe_load(open(yaml_file, 'r')), 
    args_namespace = dict2object(args_dict[0])
    return args_namespace

def prepare_namespace_args(args:SimpleNamespace):

    K = test_intrinsics_dict[args.camera_id]
    args.fx  = K[0,0]*args.scale; args.fy = K[1,1]*args.scale
    args.px = K[0,2]*args.scale; args.py = K[1,2]*args.scale
    args.K = np.array([
        [args.fx, 0, args.px], 
        [0, args.fy, args.py], 
        [0,0,1]
    ])
    args.trans_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std),
    ]) if args.normalize else transforms.ToTensor()

    return args


class BoardlessCalibrator():

    def __init__(
        self, intrinsics_dict : dict, ctrnet_args : SimpleNamespace = None, 
    ):

        self._intrinsics_dict = intrinsics_dict
        self._ctrnet_args = yaml2namespace(abspath('./r2d2/calibration/ctrnet/config.yaml'))\
            if ctrnet_args is None else ctrnet_args

        # Mesh files for differentiable renderer (useful in training stage)
        self.robot_mesh_files = [
            'link0/link0.obj', 'link1/link1.obj', 'link2/link2.obj',
            'link3/link3.obj', 'link4/link4.obj', 'link5/link5.obj',
            'link6/link6.obj', 'link7/link7.obj', 'hand/hand.obj']
        self.robot_mesh_files = [join(expanduser(ctrnet_args.meshobj_dir), f) for f in self.robot_mesh_files]
        
        # Pre-defined 3D keypoint locations for PnP solver
        self.keypoints_3d = np.array([
            [-1.43076244e-01, -2.14468290e-03,  8.92312189e-03],   # In Robot frame
            [-1.25680570e-01,  6.31571448e-04,  5.56254291e-02],
            [-5.49979992e-02, -8.49999997e-05,  1.40000001e-01],
            [ 5.49880005e-02, -1.55699998e-03,  1.40000001e-01],
            [ 7.15439990e-02,  3.29000002e-04,  1.34900003e-03],
            [-1.09613143e-02, -8.00555708e-02,  3.85714277e-03],
            [-1.10056000e-02,  8.00555708e-02,  3.85714277e-03]], dtype=np.float32)
        
        self.model = CtRNetBaseOnly(ctrnet_args)
        self.model.eval()

    def calibrate(self, cam_id, dataloader, summary_writer=None):
        return self._calibrate_cam_to_base(cam_id, dataloader, summary_writer)

    def _calibrate_cam_to_base(self, cam_id, dataloader, summary_writer=None):
        assert cam_id in self._intrinsics_dict.keys(), f'Camera {cam_id} not found in intrinsics dictionary'

        robot_renderer = self.model.setup_robot_renderer(self.robot_mesh_files[:3])
        
        kp3d = torch.from_numpy(self.keypoints_3d).to(self.model.device, dtype=torch.float32)
        kp3d = kp3d.repeat(dataloader.batch_size, 1, 1)
        K = torch.from_numpy(args.K).to(device=self.model.device, dtype=torch.float32)

        ctr_list = []
        with torch.no_grad():
            for batch_id, (img, joint_angle, extrinsic) in enumerate(pbar:=tqdm(dataloader)):
                img, joint_angle, extrinsic = to_device(img, joint_angle, extrinsic)
                ctr, kp2d, seg = self.model.inference_batch_images_base_only(img, kp3d)

                ctr_list.append(ctr)
                ctr_all = torch.cat(ctr_list, dim=0)
                t_avg = ctr_all[:,:3].mean(dim=0)
                r_avg = average_euler_angles(ctr_all[:,3:])
                ctr_avg = torch.cat([t_avg, r_avg])

                pbar.set_postfix({'ctr shape': ctr.shape})
                if batch_id % 4 == 0:
                    pose_and_seg_pred = plot_pose_and_gtkp(ctr_avg, K, kp3d[0], kp2d[0], None, img[0], seg[0], fname=None, title='Predicted pose')
                    robot_mesh = robot_renderer.get_robot_mesh(joint_angle[0].detach().cpu().numpy())
                    img_rendered = self.model.render_single_robot_mask(ctr_avg, robot_mesh, robot_renderer)
                    
                    if summary_writer is None: continue
                    summary_writer.add_image("eval_img/masknpose_pred", torch.cat([torch.tensor(pose_and_seg_pred).permute(2,0,1)] ,dim=-1), global_step=batch_id)
                    summary_writer.add_image("eval_img/seg_pred|render", torch.cat([
                        seg[0].repeat(3,1,1),
                        img_rendered.repeat(3,1,1)
                    ], dim=-1), global_step=batch_id)
        return ctr_avg  # [t_vec, r_vec]


if __name__ == "__main__":

    print('\n====================================================================================')
    print(f'Testing BoardlessCalibrator...')
    print(f"Before proceeding, please do the following:\
        \n\t1. Configure the <data_folder> attribute in './r2d2/calibration/ctrnet/config.yaml'\
        \n\t2. Build the environment by running `conda env create --file ./r2d2/calibration/ctrnet/environment.yaml`\
        \n\t3. Make sure the following intrinsic table matches your camera id and camera intrinsics")
    print('====================================================================================\n')
    
    test_intrinsics_dict = {
        '23404442_left': np.array([[522.5022583, 0. , 640.49182129], [0., 522.5022583, 353.23074341],[0.,0.,1.]]),
        '23404442_right': np.array([[522.5022583, 0. , 640.49182129], [0., 522.5022583, 353.23074341],[0.,0.,1.]]),
        '29838012_left': np.array([[523.22283936, 0. , 639.23681641], [0., 523.22283936, 352.50140381],[0.,0.,1.]]),
        '29838012_right': np.array([[523.22283936, 0. , 639.23681641], [0., 523.22283936, 352.50140381],[0.,0.,1.]]),
        '19824535_left': np.array([[697.90771484, 0. , 649.59765625], [0., 697.90771484, 354.90002441],[0.,0.,1.]]),
        '19824535_right': np.array([[697.90771484, 0. , 649.59765625], [0., 697.90771484, 354.90002441],[0.,0.,1.]])
    }

    try: # If running this file via `python calibration_utils_boardless.py`
        args = yaml2namespace(join(os.getcwd(), './ctrnet/config.yaml')) 
    except FileNotFoundError: # If running this file via VSCode debugger
        args = yaml2namespace(join(os.getcwd(), './r2d2/calibration/ctrnet/config.yaml')) 
    
    args = prepare_namespace_args(args)
    summary_writer = SummaryWriter(abspath(args.log_dir))
    print(f'Summary writer log dir: {summary_writer.log_dir}')
    
    dataset = R2D2DatasetBlock(
        data_folder =       args.data_folder, 
        camera_id =         args.camera_id, 
        trans_to_tensor =   args.trans_to_tensor,
        n_kp =              args.n_kp,
        scale =             args.scale) 
    indices = np.arange(len(dataset))[::len(dataset)//args.num_samples_per_trajecory+1]
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, 
        batch_size =    args.batch_size, 
        num_workers =   args.num_workers,
        shuffle =       args.shuffle
    )

    calibrator = BoardlessCalibrator(test_intrinsics_dict, args)
    ctr = calibrator.calibrate(args.camera_id, dataloader, summary_writer)
    print(f'Translation: {ctr[:3]}, Rotation: {ctr[3:]}')
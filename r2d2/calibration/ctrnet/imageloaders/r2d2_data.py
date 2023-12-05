# Author        : Kaiyuan Wang
# Affiliation   : ARCLab @ UC San Diego
# Email         : k5wang@ucsd.edu
# Created on    : Mon Aug 21 2023


'''Residential Robot Demonstration Dataset (R2D2) data loader. '''

import sys, torch, glob, os, cv2
import numpy as np, pandas as pd
import torchvision.transforms as transforms
from os.path import join, abspath, expanduser, isdir, exists, dirname
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from PIL import Image
from tqdm import tqdm

BASE_DIR = abspath(join(dirname(__file__), '../..'))
sys.path.append(BASE_DIR)
from r2d2.trajectory_utils.misc import load_trajectory

INTRINSICS_TABLE = {
    '23404442_left': np.array([[522.5022583, 0. , 640.49182129], [0., 522.5022583, 353.23074341],[0.,0.,1.]]),
    '23404442_right': np.array([[522.5022583, 0. , 640.49182129], [0., 522.5022583, 353.23074341],[0.,0.,1.]]),
    '29838012_left': np.array([[523.22283936, 0. , 639.23681641], [0., 523.22283936, 352.50140381],[0.,0.,1.]]),
    '29838012_right': np.array([[523.22283936, 0. , 639.23681641], [0., 523.22283936, 352.50140381],[0.,0.,1.]]),
    '19824535_left': np.array([[697.90771484, 0. , 649.59765625], [0., 697.90771484, 354.90002441],[0.,0.,1.]]),
    '19824535_right': np.array([[697.90771484, 0. , 649.59765625], [0., 697.90771484, 354.90002441],[0.,0.,1.]])
}

DIST_COEFFS_TABLE = {
    '23404442' : np.array([0., 0., 0., 0., 0.]),
    '29838012' : np.array([0., 0., 0., 0., 0.]),
    '19824535' : np.array([0., 0., 0., 0., 0.])
}

PANDA_MESH_FILES = [BASE_DIR + "/ctrnet/urdfs/Panda/meshes/visual/link0/link0.obj",
                BASE_DIR + "/ctrnet/urdfs/Panda/meshes/visual/link1/link1.obj",
                BASE_DIR + "/ctrnet/urdfs/Panda/meshes/visual/link2/link2.obj",
                BASE_DIR + "/ctrnet/urdfs/Panda/meshes/visual/link3/link3.obj",
                BASE_DIR + "/ctrnet/urdfs/Panda/meshes/visual/link4/link4.obj",
                BASE_DIR + "/ctrnet/urdfs/Panda/meshes/visual/link5/link5.obj",
                BASE_DIR + "/ctrnet/urdfs/Panda/meshes/visual/link6/link6.obj",
                BASE_DIR + "/ctrnet/urdfs/Panda/meshes/visual/link7/link7.obj",
                BASE_DIR + "/ctrnet/urdfs/Panda/meshes/visual/hand/hand.obj"]


# TODO: Generalize kp 2d and 3d to all cameras
KEYPOINTS_2D = {
    '23404442_left': np.array([[215.06241, 447.73056], [221.11848, 414.4581 ], [251.85588, 367.2945 ],
                [345.09045, 360.57587], [400.59467, 448.07397], [306.23264, 498.95663], [np.nan, np.nan]], dtype=np.float32),
    '23404442_right': None,
    "29838012_left": np.array([[1097.1947, 569.90356], [1096.2913, 539.73474], [1072.9938, 480.52173],
                [976.2365, 452.03995], [921.3841, 526.7741], [np.nan, np.nan], [1005.298, 583.37695]], dtype=np.float32),
    '29838012_right': None
}

KEYPOINTS_3D = np.array([[-1.43076244e-01, -2.14468290e-03,  8.92312189e-03],   # In Robot frame
                        [-1.25680570e-01,  6.31571448e-04,  5.56254291e-02],
                        [-5.49979992e-02, -8.49999997e-05,  1.40000001e-01],
                        [ 5.49880005e-02, -1.55699998e-03,  1.40000001e-01],
                        [ 7.15439990e-02,  3.29000002e-04,  1.34900003e-03],
                        [-1.09613143e-02, -8.00555708e-02,  3.85714277e-03],
                        [-1.10056000e-02,  8.00555708e-02,  3.85714277e-03]], dtype=np.float32)

def get_kp(camera_id=None, type='2d'): 
    assert type in ['2d', '3d'], f"Keypoint type [{type}] not recognized. Please choose from ['2d', '3d']"
    if camera_id is None or type == '3d': return KEYPOINTS_3D
    else: return KEYPOINTS_2D[camera_id]

def get_kp2ds(kp2d_label_dirs, width=1280, n_kp=7, fname_csv='CollectedData_kp2d.csv'):
    ''' Read .csv file and extract 2D keypoints annotation result 
    Inputs:
        - kp2d_label_dirs: List of directories containing .csv files
        - n_kp: number of keypoints
        - fname_csv: Name of csv file containing 2D keypoint annotations
    Returns:
        - kp2d_data: Dictionary with camera_id keys and 2D keypoints values
    
    Note: Columns in CSV files are formatted as:
        <irrelevant cols>, kp1_left_x, kp1_left_y,kp2_left_x, kp2_left_y, ...,
                           kp1_right_x, kp1_right_y,kp2_right_x, kp2_right_y, ... 
    '''

    kp2d_data = {}
    for path in kp2d_label_dirs:
        csv_path = join(path, fname_csv)
        camera_id = path.split('/')[-1]

        df = pd.read_csv(csv_path)
        max_cols = n_kp*4   # TODO: Expand max cols to args.n_kp * 4 to extract keypoints for right images
        kp2d = df.iloc[2:,1:max_cols+1].to_numpy(dtype=np.float32)
        kp2d_left, kp2d_right = kp2d[:,:max_cols//2], kp2d[:,max_cols//2:] 
        
        kp2d_data[f'{camera_id}_left'] = np.mean(kp2d_left,axis=0).reshape(-1,2)
        kp2d_data[f'{camera_id}_right'] = np.mean(kp2d_right,axis=0).reshape(-1,2)
        kp2d_data[f'{camera_id}_right'][:,0] -= width
        
    return kp2d_data

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)



class R2D2DatasetBlock(Dataset):

    def __init__(self, data_folder='~/Desktop/data/r2d2_household/pen_out_several/Fri_Apr_21_10_35_02_2023',
        camera_id='23404442_left', scale=1., trans_to_tensor=None, n_kp=7, kp2ds_info_dict:dict=None
    ):
        ''' A block of R2D2 dataset. Concat all R2D2DatasetBlocks to construct full R2D2 dataset.
        This class can be used as a stand-alone dataset by itself.

        Inputs:
            - data_folder:      path to data dir, e.g. '~/data/pen_in/Fri_Apr_21_10_42_22_*'
            - scale: tuple      H, W scale factor for image size
            - trans_to_tensor:  torchvision.transforms.Compose object
            - camera_id:        a list of camera_id's selected from
                                {23404442_left', '23404442_right', '29838012_left','29838012_right'}
            - kp2ds_info_dict:  Dictionary with 2D keypoints values for the specific camera_id
              NOTE: '19824535_left', '19824535_right are excluded because they locate on the gripper
        '''
        if trans_to_tensor is None:
            self.trans_to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else: self.trans_to_tensor = trans_to_tensor

        self.scale = scale
        self.n_kp = n_kp
        self.camera_id = camera_id
        self.kp2ds_info_dict = kp2ds_info_dict

        self.data_folder = abspath(expanduser(data_folder))
        self.h5_path = join(self.data_folder, "trajectory.h5")
        self.mp4_path = join(self.data_folder, "recordings/MP4")
        self.timestep_list = load_trajectory(filepath=self.h5_path, recording_folderpath=self.mp4_path)

        # list of (time, camera_id) pairs
        # self.data_ids = list(itertools.product(range(len(self.timestep_list)), self.camera_ids))
        img_sample = self.timestep_list[0]['observation']['image'][self.camera_id]
        self.height = int(img_sample.shape[0] * self.scale)
        self.width = int(img_sample.shape[1] * self.scale)

        if self.kp2ds_info_dict is not None:
            self.kp2d_gt = torch.tensor(self.kp2ds_info_dict['kp2d'], dtype=torch.float32) * self.scale
        else:
            try:
                kp2d_label_dir = join(self.mp4_path, list(filter(lambda path:'.'not in path, os.listdir(self.mp4_path)))[0], 'labeled-data')
                self.kp2d_gt = get_kp2ds(
                    [join(kp2d_label_dir, path) for path in list(filter(lambda path:'_'not in path, os.listdir(kp2d_label_dir)))],
                    width=img_sample.shape[1], n_kp=self.n_kp
                )
                self.kp2d_gt = torch.tensor(self.kp2d_gt[self.camera_id] * self.scale, dtype=torch.float32)
            except IndexError:
                print(f"WARNING: No 2D keypoint annotations found for camera [{self.camera_id}].")
                self.kp2d_gt = None

    def __len__(self):  return len(self.timestep_list)

    def __getitem__(self, idx):
        ''' Returns the following:
            - img:          (3,H,W) torch.Tensor
            - joint_angle:  (7) torch.Tensor
            - self.kp2d_gt: (self.n_kp,2) torch.Tensor
            - extrinsic:    (6,) torch.Tensor. Rotation, translation.
        '''
        # (time, camera_id) = self.data_ids[idx]
        img = self.timestep_list[idx]['observation']['image'][self.camera_id] # dict: camera_id -> img
        img = Image.fromarray(img) # (H,W,3) uint8
        img = img.resize((self.width, self.height))
        img = self.trans_to_tensor(img)                 # (3,H,W)
        joint_angle = self.timestep_list[idx]['action']['joint_position']
        joint_angle = torch.tensor(joint_angle, dtype=torch.float32)
        extrinsic = torch.tensor(self.timestep_list[idx]['observation']['camera_extrinsics'][self.camera_id], dtype=torch.float32)
        
        if self.kp2d_gt is None: return img, joint_angle, extrinsic
        else:                    return img, joint_angle, extrinsic, self.kp2d_gt

class R2D2DatasetBlockWithSeg(R2D2DatasetBlock):

    def __init__(self, data_folder='~/Desktop/data/r2d2_household/pen_out_several/Fri_Apr_21_10_35_02_2023',
        camera_id='23404442_left', scale=1., trans_to_tensor=None, n_kp=7, kp2ds_info_dict:dict=None
    ):
        super().__init__(data_folder, camera_id, scale, trans_to_tensor, n_kp, kp2ds_info_dict=kp2ds_info_dict)
        self.seg_path = join(self.data_folder, f"{self.camera_id}_seg_3joints.npz")
        self.seg_masks = np.load(self.seg_path)['masks']
    
    def __getitem__(self, idx):
        if self.kp2d_gt is None:
            img, joint_angle, extrinsic = super().__getitem__(idx)
        else:
            img, joint_angle, extrinsic, kp2d_gt = super().__getitem__(idx)

        seg = cv2.resize(self.seg_masks[idx].astype(np.uint8), (self.width, self.height))
        seg = torch.tensor(seg)
        return img, joint_angle, kp2d_gt, extrinsic, seg

class R2D2StereoDatasetBlock(R2D2DatasetBlock):

    def __getitem__(self, idx):
        cam_id_base = self.camera_id[:-5]
        img_l = self.timestep_list[idx]['observation']['image'][f'{cam_id_base}_left'] 
        img_r = self.timestep_list[idx]['observation']['image'][f'{cam_id_base}_right'] 
        img_l, img_r = Image.fromarray(img_l), Image.fromarray(img_r)
        img_l, img_r = img_l.resize((self.width, self.height)), img_r.resize((self.width, self.height))
        img_l, img_r = self.trans_to_tensor(img_l), self.trans_to_tensor(img_r)

        joint_angle = self.timestep_list[idx]['action']['joint_position']
        joint_angle = torch.tensor(joint_angle, dtype=torch.float)

        img = torch.cat((img_l, img_r), dim=1)  # Stack left and right images along X axis

        return img, joint_angle

def get_camera_intrinsics(camera_id):
    ''' Get camera (3, 3) intrinsics for R2D2 dataset.'''

    assert camera_id in ['23404442_left', '23404442_right', '29838012_left','29838012_right', '19824535_left', '19824535_right'], \
        f"camera_id [{camera_id}] not recognized. Please choose from ['23404442_left', '23404442_right', '29838012_left','29838012_right', '19824535_left', '19824535_right']"

    return INTRINSICS_TABLE[camera_id]

def get_robot_mesh_files():
    ''' Get PANDA robot mesh files for R2D2 dataset.'''
    return PANDA_MESH_FILES

def get_r2d2_dataset(data_folder='~/Desktop/data/r2d2_household/pen_out_several',
    camera_ids=['23404442_left', '23404442_right', '29838012_left','29838012_right'],
    max_num_blocks=2, scale=(2/3, 1/2), trans_to_tensor=None
):
    '''
    Inputs:
        ...
        - max_num_blocks: maximum number of blocks (i.e. number of dirs) to load.
    '''

    paths_to_blocks = sorted([path for path \
        in glob.glob(join(abspath(expanduser(data_folder)), '*')) if isdir(path)])

    max_num_blocks = min(max_num_blocks, len(paths_to_blocks))
    paths_to_blocks = paths_to_blocks[:max_num_blocks]

    print(f'\nConstructing dataset blocks. This might take a while...\n')
    dataset_blocks = []
    for path in (pbar:=tqdm(paths_to_blocks, desc="Constructing dataset blocks")):
        dataset_blocks.append(R2D2DatasetBlock(
            path, camera_ids=camera_ids, scale=scale,trans_to_tensor=trans_to_tensor))
        pbar.set_postfix({"block length": len(dataset_blocks[-1])})
    pbar.close()

    return ConcatDataset(dataset_blocks)



if __name__ == "__main__":

    print(" Executing test run of r2d2_data.py ... \n")

    data_dir ='~/Desktop/data/r2d2_household/pen_out_several'
    if not exists(expanduser(data_dir)):
        raise ValueError(f"Data directory [{data_dir}] does not exist. Please change file path in main function of r2d2_data.py")
    dataset = get_r2d2_dataset(data_dir, max_num_blocks=1)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    print(f"Dataset length: [{len(dataset)}]\n")
    for i, batch in (pbar1:=tqdm(enumerate(data_loader), desc="Enumerating dataset")):
        img, joint_angle = batch
        pbar1.set_description(f"Loading batch {i} out of {len(data_loader)}")
        pbar1.set_postfix({"img_shape": img.shape, "joint_angle_shape": joint_angle.shape})
    pbar1.close()
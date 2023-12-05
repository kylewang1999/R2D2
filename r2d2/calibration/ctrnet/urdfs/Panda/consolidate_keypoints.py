'''
File Created:  2023-Sep-22nd Fri 12:24
Author:        Kaiyuan Wang (k5wang@ucsd.edu)
Affiliation:   ARCLab @ UCSD
Description:   Read-in hand-annotated keypoint .ply files, consolidate them 
               into a single PointCloud and save as keypoints.ply
NOTE on each keypoint .ply file
    - Each file is annotated using mesh lab. 
    - Each file might contain 1 or more points.
        - If more than 1 point in a file, take the mean of all points to consolidate into a single point.
'''

from os.path import join
import open3d as o3d, numpy as np

# mesh = o3d.io.read_triangle_mesh("/Users/KyleWang/Desktop/Panda/meshes/visual/link0/link0.obj")
# scp -P40 /Users/KyleWang/Desktop/Panda/meshes/visual/link0/ kyle@66.27.66.217:~/Desktop/r2d2/ctrnet

ply_dir = '/Users/KyleWang/Desktop/Panda/meshes/visual/link0'
ply_kp_files = [join(ply_dir, f'link0_kp{i+1}.ply') for i in range(7)]

keypoints = [np.asarray(o3d.io.read_point_cloud(ply_file).points) for ply_file in ply_kp_files]
keypoints = np.array([np.mean(pcd, axis=0) for pcd in keypoints])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(keypoints)
o3d.io.write_point_cloud(join(ply_dir, 'keypoints.ply'), pcd)

print(f'Wrote keypoints of shape {keypoints.shape} to {join(ply_dir, "keypoints.ply")}')
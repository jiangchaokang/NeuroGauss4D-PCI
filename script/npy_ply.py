import os
import numpy as np

def save_pointcloud_as_ply(tensor, rgb, save_path):
    points = tensor
    ply_header = """ply
        format ascii 1.0
        element vertex {}
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        """.format(points.shape[0])
    points_with_rgb = np.concatenate((points, np.tile(np.array(rgb, dtype=np.uint8), (points.shape[0], 1))), axis=1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as ply_file:
        ply_file.write(ply_header)
        np.savetxt(ply_file, points_with_rgb, fmt="%f %f %f %d %d %d")
    
    print(f"Point cloud saved as {save_path}")

npy_folder_path = './NeuroGauss4D/log/visualizations/dense_pointclouds/02_dense_sf/'


for root, dirs, files in os.walk(npy_folder_path):
    for filename in files:
        if filename.endswith('.npy'):
            npy_file_path = os.path.join(root, filename)
            ply_file_path = os.path.join(root, os.path.splitext(filename)[0] + '.ply')
            tensor = np.load(npy_file_path)
            if "gt" in filename:
                save_pointcloud_as_ply(tensor, (0,233,0), ply_file_path)
            else:
                save_pointcloud_as_ply(tensor, (233,0,0), ply_file_path)
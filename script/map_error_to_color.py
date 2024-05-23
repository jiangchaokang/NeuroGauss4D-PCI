# import numpy as np
# import sys
# sys.path.append('./utils/EMD')
# sys.path.append('./utils/CD')
# from emd import earth_mover_distance
# import chamfer3D.dist_chamfer_3D
# import os
# import plyfile
# import pdb
# import torch

# chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()

# def map_error_to_color(error, min_val, max_val):

#     if error < min_val:
#         return (0, 1, 1)  
#     elif error > max_val:
#         return (1, 0, 0)  
#     else:
#         r = (error - min_val) / (max_val - min_val)
#         g = 1 - r
#         b = 1 - r
#         return (r, g, b)


# def save_color_ply(tensor, rgb, save_path):
#     points = tensor
#     ply_header = """ply
#         format ascii 1.0
#         element vertex {}
#         property float x
#         property float y
#         property float z
#         property uchar red
#         property uchar green
#         property uchar blue
#         end_header
#         """.format(points.shape[0])
#     points_with_rgb = np.concatenate((points, np.array(rgb)*255, ), axis=1)

#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     with open(save_path, 'w') as ply_file:
#         ply_file.write(ply_header)
#         np.savetxt(ply_file, points_with_rgb, fmt="%f %f %f %d %d %d")
    
#     print(f"Point cloud saved as {save_path}")


# # pred_cloud_path = './NeuroGauss4Dlog/visualizations/pred_other/NeuralPCI/longdress/027/pc_intp_1.npy'
# pred_cloud_path = './NeuroGauss4Dlog/visualizations/pred_other/3DSFLabelling/soldier035/pc_intp_1.npy'
# gt_cloud_path = './NeuroGauss4Dlog/visualizations/pred_our/soldier/035/gt_1.npy'
# pred_cloud = np.load(pred_cloud_path)
# gt_cloud = np.load(gt_cloud_path)
# pred_cloud_tensor = torch.tensor(pred_cloud).unsqueeze(0).cuda()
# gt_cloud_tensor = torch.tensor(gt_cloud).unsqueeze(0).cuda()

# dist1, dist2, _, _ = chamLoss(pred_cloud_tensor, gt_cloud_tensor)
# dist1, dist2 = dist1.squeeze(0).cpu().numpy(), dist2.squeeze(0).cpu().numpy()
# # dist = earth_mover_distance(pred_cloud_tensor, gt_cloud_tensor, transpose=False)
# # emd_error = (dist / pred_cloud_tensor[0].shape[0]) #.mean()


# dist1_colors = []
# dist2_colors = []
# # pdb.set_trace()
# for cd_error, emd_error in zip(dist1, dist2):
#     cd_error = min(max(cd_error, 0), 0.002)
#     cd_color = map_error_to_color(cd_error, 0, 0.002)
#     dist1_colors.append(cd_color)

#     emd_error = min(max(emd_error, 0), 0, 0.001)
#     emd_color = map_error_to_color(emd_error, 0, 0.001)
#     dist2_colors.append(emd_color)

# filename = os.path.basename(pred_cloud_path)
# filename_without_ext = os.path.splitext(filename)[0]

# cd_colors_path = os.path.join(os.path.dirname(pred_cloud_path), 'CD1_' + filename_without_ext + '.npy')
# emd_colors_path = os.path.join(os.path.dirname(pred_cloud_path), 'CD2_' + filename_without_ext + '.npy')
# np.save(cd_colors_path, np.array(dist1_colors))
# np.save(emd_colors_path, np.array(dist2_colors))

# cd_ply_path = os.path.join(os.path.dirname(pred_cloud_path), 'CD1_' + filename_without_ext + '.ply')
# emd_ply_path = os.path.join(os.path.dirname(pred_cloud_path), 'CD2_' + filename_without_ext + '.ply')
# save_color_ply(pred_cloud, dist1_colors, cd_ply_path)
# save_color_ply(pred_cloud, dist2_colors, emd_ply_path)

import cv2
import numpy as np

def map_error_to_color(error, min_val, max_val):
    if error < min_val:
        return (0, 0, 1)  
    elif error > max_val:
        return (1, 0, 0)
    else:
        r = (error - min_val) / (max_val - min_val)
        g = 0
        b = 1 - r
        return (r, g, b)

min_error = 0.0
max_error = 10.0

img = np.zeros((100, 600, 3), dtype=np.uint8)

for col in range(img.shape[1]):
    error = min_error + (max_error - min_error) * col / (img.shape[1] - 1)
    color = map_error_to_color(error, min_error, max_error)
    color = tuple(int(c * 255) for c in color)
    img[:, col] = color

cv2.imshow('Color Band', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
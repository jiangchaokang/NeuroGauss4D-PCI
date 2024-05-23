import numpy as np
import cv2
import matplotlib.pyplot as plt
from open3d import visualization, geometry, io
import pdb
import open3d as o3d

def read_bin(bin_path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points

def read_label(label_path):
    labels = np.fromfile(label_path, dtype=np.uint32)
    return labels

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_calib(calib_path):
    with open(calib_path, 'r') as f:
        file = f.readlines()
    calib = {}
    for line in file:
        key, value = line.split(':', 1)
        calib[key] = np.array([float(x) for x in value.split()])
    P2 = calib['P2'].reshape(3, 4)
    Tr = np.vstack((calib['Tr'].reshape(3, 4), [0, 0, 0, 1]))
    return P2, Tr


def project_points(points, P2, Tr):
    N = points.shape[0]
    points_hom = np.hstack((points[:, :3], np.ones((N, 1)))) 
    points_cam = Tr @ points_hom.T  
    points_img = P2 @ points_cam
    points_img /= points_img[2, :] 

    mask = (points_cam[2, :] > 0) & (points_img[0, :] >= 0) & (points_img[0, :] < 1242) & (points_img[1, :] >= 0) & (points_img[1, :] < 375)
    return points_img[:2, mask].T, mask  

def draw_large_point(image, point, color, size=3):
    x, y = int(point[0]), int(point[1])
    half_size = size // 2  

    x_min = max(x - half_size, 0)
    x_max = min(x + half_size + 1, image.shape[1])
    y_min = max(y - half_size, 0)
    y_max = min(y + half_size + 1, image.shape[0])

    image[y_min:y_max, x_min:x_max] = color[:3]

def visualize_projection(image, points_img, points, labels):
    image_copy = np.copy(image)
    label_colors = {
        0: [255, 255, 255],   
        10: [0, 255, 0],      #
        40: [0, 0, 255],      #
        44: [255, 255, 0],    #
        48: [0, 255, 255],    #
        49: [255, 0, 255],    #
        50: [192, 192, 192],  #
        51: [128, 0, 0],      #
        70: [128, 128, 0],    #
        71: [0, 128, 0],      #
        72: [128, 0, 128],    #
        80: [0, 128, 128],    #
        81: [0, 0, 128],      
        99: [128, 128, 128],  
        254: [64, 64, 64]     
    }
    default_color = [75, 0, 192]
    colors = [label_colors.get(label & 0xFFFF, default_color) for label in labels]
    
    # cmap = plt.cm.get_cmap('jet', 256)
    # colors = cmap(points[:, 3] / np.max(points[:, 3]))

    # for p, color in zip(points_img, colors):
    #     x, y = int(p[0]), int(p[1])
    #     if 0 <= x < image_copy.shape[1] and 0 <= y < image_copy.shape[0]:  
    #         image_copy[y, x, :] = color[:3] #* 255  

    for p, color in zip(points_img, colors):
        draw_large_point(image_copy, p, color, size=2)

    save_path = "./NeuroGauss4D/log/visualizations/kitti/rgb08/000321_projection.png"
    plt.imsave(save_path, image_copy.astype(np.uint8))  

def color_points_by_label(points, labels):
    label_colors = {
        0: [255, 0, 0],       
        10: [0, 255, 0],      
        40: [0, 0, 255],      
        44: [255, 255, 0],    
        48: [0, 255, 255],    
        49: [255, 0, 255],    
        50: [192, 192, 192],  
        51: [128, 0, 0],      
        70: [128, 128, 0],    
        71: [0, 128, 0],      
        72: [128, 0, 128],    
        80: [0, 128, 128],    
        81: [0, 0, 128],      
        99: [128, 128, 128],  
        254: [64, 64, 64]     
    }

    
    default_color = [100, 100, 100]  

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    colors = [label_colors.get(label & 0xFFFF, default_color) for label in labels]
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors) / 255.0)  

    return pcd

def save_ply(pcd, filename):
    io.write_point_cloud(filename, pcd)

def main():
    bin_path = './DATASETS/Driving_datasets/odKITTI/Raw_od_LIDAR/dataset/sequences/00/velodyne/000321.bin'
    label_path = './DATASETS/Driving_datasets/odKITTI/Raw_od_LIDAR/dataset/sequences/00/labels/000321.label'
    image_path = './DATASETS/Driving_datasets/odKITTI/Raw_od_LIDAR/dataset/sequences/02/image_2/000321.png'
    calib_path = './DATASETS/Driving_datasets/odKITTI/Raw_od_LIDAR/dataset/sequences/00/calib.txt'
    

    points = read_bin(bin_path)
    labels = read_label(label_path)
    
    image = load_image(image_path)
    P2, Tr = load_calib(calib_path)
    
    points_img, mask = project_points(points, P2, Tr)
    points_visible = points[mask]  
    labels_visible = labels[mask]  
    
    # points_visible = np.load("./NeuroGauss4D/log/visualizations/kitti/rgb00/pred_time_pc2_v1.npy")

    # np.save("./NeuroGauss4D/log/visualizations/kitti/rgb00/000321.npy", points_visible[:,:3])
    # pdb.set_trace()
    
    visualize_projection(image, points_img, points_visible, labels_visible)

    pcd = color_points_by_label(points_visible, labels_visible)
    save_ply(pcd, "./NeuroGauss4D/log/visualizations/kitti/rgb08/000321_semantic_pointcloud.ply")

if __name__ =="__main__":
    main()
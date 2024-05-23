import numpy as np
from scipy.spatial import KDTree

def remove_redundant_points(point_cloud, radius=0.009):
    tree = KDTree(point_cloud)
    keep_mask = np.ones(len(point_cloud), dtype=bool)
    
    for i, point in enumerate(point_cloud):
        if keep_mask[i]:
            neighbors = tree.query_ball_point(point, radius)
            if len(neighbors) > 2:
                keep_mask[neighbors[2:]] = False
    
    return point_cloud[keep_mask]

tensor = np.load("./NeuroGauss4Dlog/visualizations/dense_pointclouds/Test_dense.npy")
pc = remove_redundant_points(tensor)
np.save("./NeuroGauss4Dlog/visualizations/dense_pointclouds/Test_spare.npy", pc)
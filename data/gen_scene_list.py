# import os
# from glob import glob
# import pdb

# data_path = './Driving_datasets/PCI/NL-Drive/NL-Drive/interval1/kitti_wground/00/velodyne'
# output_file = './Driving_datasets/PCI/NL-Drive/NL-Drive/interval1/kitti_wground/scene_list.txt'
# interval_frames = 5
# bin_files = sorted(glob(os.path.join(data_path, '*.bin')))

# with open(output_file, 'w') as f:
#     pass

# with open(output_file, 'a') as f:
#     for i in range(0, len(bin_files), 4 * interval_frames):
#         frame1, frame2, frame3, frame4 = [os.path.basename(bin_files[i + j * interval_frames]) for j in range(4)]
#         intermediate_frames = [os.path.basename(bin_files[i+5 + (j + 1)]) for j in range(interval_frames - 1)]
#         combination = f"{frame1} {frame2} {frame3} {frame4} " + " ".join(intermediate_frames)
#         f.write(combination + '\n')



import os
from glob import glob
data_path = './Driving_datasets/PCI/NL-Drive/NL-Drive/interval1/kitti_wground/00/velodyne'
output_file = './Driving_datasets/PCI/NL-Drive/NL-Drive/interval1/kitti_wground/scene_list_interval1.txt'

bin_files = sorted(glob(os.path.join(data_path, '*.bin')))

with open(output_file, 'w') as f:
    pass

with open(output_file, 'a') as f:
    for i in range(0, len(bin_files), 5):
        frame1, frame2, frame3, frame4, frame5 = [os.path.basename(path) for path in bin_files[i:i+5]]
        combination = f"{frame1} {frame2} {frame4} {frame5} {frame3}"
        f.write(combination + '\n')
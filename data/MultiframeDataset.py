import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import open3d

# import cv2
import torch.utils.data
from glob import glob
import os.path as osp
# from .transforms import ProcessData
# from PIL import Image

class NLDriveDataset(Dataset):
    """
    Args:
        data_root: path for NL-Drive dataset
        scene_list: path of point cloud sequence list to load samples
        interval: point cloud sequence downsampling interval, pick one frame from every (interval) frames,
                  i.e. (interval - 1) interpolation frame between every two frames [default: 4]
        num_points: sample a fixed number of points in each input and gt point cloud frame [default: 8192]
        num_frames: number of input point cloud frames [default: 4]
    """
    def __init__(self, data_root, scene_list, interval=4, num_points=8192, num_frames=4):
        super(NLDriveDataset, self).__init__()
        self.data_root = data_root
        self.num_points = num_points
        self.scene_list = scene_list
        self.interval = interval
        self.num_frames = num_frames
        self.velodynes = self.read_scene_list()
    
    def read_scene_list(self):
        velodynes = []
        with open(self.scene_list, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n').split(' ')
                velodynes.append(line)
        return velodynes

    def __getitem__(self, index):
        sample_names = self.velodynes[index]
        pc = []
        gt = []
        pc_points_idx = []
        gt_points_idx = []

        for i in range(self.num_frames):
            # load data
            pc_path = os.path.join(self.data_root, sample_names[i])
            # print(pc_path)
            pc_raw = np.fromfile(pc_path, dtype = np.float32, count = -1).reshape([-1, 3])
            # pc_raw = np.fromfile(pc_path, dtype = np.float32, count = -1).reshape([-1, 4])[:,:3]
            # pc_raw = self.remove_outlier(pc_raw)
            pc.append(pc_raw)

            # sample n_points
            num = pc_raw.shape[0]
            if num >= self.num_points:
                pc_points_idx.append(np.random.choice(num, self.num_points, replace=False))
            else:
                pc_points_idx.append(np.concatenate((np.arange(num), np.random.choice(num, self.num_points - num, replace=True)), axis = -1))
        
        num_gt = len(sample_names) - self.num_frames
        gt_intv = num_gt // (self.interval - 1)
        for i in range(self.interval-1):
            # load data and rm_ground if needed
            gt_path = os.path.join(self.data_root, sample_names[3 + (i+1)*gt_intv])
            # print(gt_path)
            gt_raw = np.fromfile(gt_path, dtype = np.float32, count = -1).reshape([-1, 3])
            # gt_raw = np.fromfile(gt_path, dtype = np.float32, count = -1).reshape([-1, 4])[:,:3]
            # gt_raw = self.remove_outlier(gt_raw)
            gt.append(gt_raw)

            # sample n_points
            num = gt_raw.shape[0]
            if num >= self.num_points:
                gt_points_idx.append(np.random.choice(num, self.num_points, replace=False))
            else:
                gt_points_idx.append(np.concatenate((np.arange(num), np.random.choice(num, self.num_points - num, replace=True)), axis = -1))

        pc_sampled = []
        gt_sampled = []

        for i in range(self.num_frames):
            pc_sampled.append(pc[i][pc_points_idx[i], :].astype('float32'))
        for i in range(self.interval-1):
            gt_sampled.append(gt[i][gt_points_idx[i], :].astype('float32'))

        input = []
        gt = []
        for pc in pc_sampled:
            input.append(torch.from_numpy(pc))
        for intp in gt_sampled:
            gt.append(torch.from_numpy(intp))
        return input, gt

    def __len__(self):
        return len(self.velodynes)

    def remove_outlier(self, pc, nb_neighbors=16, std_ratio=2.5):
        pcl = open3d.geometry.PointCloud()
        pcl.points = open3d.utility.Vector3dVector(pc)
        cl, ind = pcl.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        cl_arr = np.asarray(cl.points)
        return cl_arr



class DHBDataset(Dataset):
    """
    data_root: path for DHB dataset
    scene_list: path of point cloud sequence list to load samples
    interval: point cloud sequence downsampling interval, pick one frame from every (interval) frames,
              i.e. (interval - 1) interpolation frame between every two frames  [default: 4]
    """
    def __init__(self, data_root, scene_list, interval=4): 
        self.data_root = data_root
        self.interval = interval
        self.scene_8IVFB = ['longdress','loot', 'redandblack', 'soldier', "squat_2_fps1024_aligned", "swing_fps1024_aligned"]  # , 'squat_2', 'swing'

        self.scenes = self.read_scene_list(scene_list)
        self.total = 0
        self.dataset_dict, self.dataset_scene_len = self.make_dataset()
        
    def read_scene_list(self, scenes_list):
        # read .txt file containing train/val scene number
        scenes = []
        with open(scenes_list, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                scenes.append(line)
        return scenes
    
    def make_dataset(self):
        scene_8IVFB = ['longdress','loot', 'redandblack', 'soldier']
        dataset_dict = {}
        dataset_scene_len = {}
        scene_ini = 0
        scene_end = 0
        for scene in self.scenes:
            if scene in self.scene_8IVFB:
                dataset_dict[scene] = self.get_rich_data_8ivfb(scene)
            else:
                continue
                dataset_dict[scene] = self.get_rich_data(scene)
            sample_len = dataset_dict[scene][-1]
            scene_end += sample_len
            dataset_scene_len[scene]=[scene_ini,scene_end]
            scene_ini = scene_end
        self.total = scene_end
        return dataset_dict, dataset_scene_len # [scene_num, 3], i.e. each scene has [data_tensor,GroupIdx,sample_len]

    def get_rich_data(self, scene):
        data_tensor = torch.load( os.path.join(self.data_root,scene+'_fps1024_aligned.pt') )[:, :, :]
        GroupIdx,sample_len=self.get_one_scene_index(len(data_tensor))
        # print('scene ====',scene, 'len seq ====', len(data_tensor))
        return [data_tensor,GroupIdx,sample_len]

    def get_rich_data_8ivfb(self, scene):
        _path = './'
        data_tensor = torch.load( os.path.join(self.data_root,scene+'.pt') )[:, :, :]
        GroupIdx,sample_len=self.get_one_scene_index(len(data_tensor))
        # print('scene ====',scene, 'len seq ====', len(data_tensor))
        return [data_tensor, GroupIdx, sample_len]

    def get_one_scene_index(self, len_):
        GroupIdx={}
        GroupIdx['pc1']=[]
        GroupIdx['pc2']=[]
        GroupIdx['pc3']=[]
        GroupIdx['pc4']=[]
        for k in range(self.interval - 1):
            GroupIdx[f'gt{k}']=[]
        ini_idx = 0
        end_idx = 0
        while ini_idx + self.interval * 3 < len_:
            end_idx = ini_idx + self.interval * 3
            GroupIdx['pc1'].append(ini_idx)
            GroupIdx['pc2'].append(ini_idx + self.interval)
            GroupIdx['pc3'].append(ini_idx + self.interval * 2)
            GroupIdx['pc4'].append(ini_idx + self.interval * 3)
            for k in range(self.interval - 1):
                GroupIdx[f'gt{k}'].append(ini_idx + self.interval + k + 1)
            ini_idx += self.interval
        assert(len(GroupIdx['pc1'])==len(GroupIdx['pc2']))
        assert(len(GroupIdx['pc1'])==len(GroupIdx['pc3']))
        assert(len(GroupIdx['pc1'])==len(GroupIdx['pc4']))
        assert(len(GroupIdx['pc1'])==len(GroupIdx['gt0']))
        sample_len = len(GroupIdx['pc1'])
        return GroupIdx, sample_len


    def pc_normalize(self, pc, max_for_the_seq):
        pc = pc.numpy()
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = max_for_the_seq
        pc = pc / m
        return pc

    def __getitem__(self, index):
        sample={}
        sample['indices']=[]

        for scene, ini_end in self.dataset_scene_len.items():
            if index < ini_end[1]:
                [data_tensor,GroupIdx,sample_len] = self.dataset_dict[scene]
                sample['scene']= scene
                inside_idx = index-ini_end[0]
                for pos, scene_sample_idx in GroupIdx.items():        
                    sample_idx = scene_sample_idx[inside_idx] # list of sample_idx
                    sample['indices'].append(sample_idx) #filled indices
                    pc = data_tensor[sample_idx]
                    if sample['scene'] in self.scene_8IVFB:
                        pc = self.pc_normalize(pc, max_for_the_seq=583.1497484423953)
                        pc = torch.from_numpy(pc)
                    sample[pos] = pc #filled pc
                sample['indices'] = np.array(sample['indices'])
                
                pc1 = sample["pc1"] # [1024, 3]
                pc2 = sample["pc2"]
                pc3 = sample["pc3"]
                pc4 = sample["pc4"]
                input = [pc1,pc2,pc3,pc4]
                gt = []
                for i in range(self.interval-1):
                    gt.append(sample[f'gt{i}'])

                return input, gt
                
    def __len__(self):
        return self.total


class ProcessData(object):
    def __init__(self, data_process_args, num_points, allow_less_points):
        self.DEPTH_THRESHOLD = data_process_args['DEPTH_THRESHOLD']
        self.no_corr = data_process_args['NO_CORR']
        self.num_points = num_points
        self.allow_less_points = allow_less_points

    def __call__(self, data):
        pc1, pc2 = data
        if pc1 is None:
            return None, None, None,

        sf = pc2[:, :3] - pc1[:, :3]

        if self.DEPTH_THRESHOLD > 0:
            near_mask = np.logical_and(pc1[:, 2] < self.DEPTH_THRESHOLD, pc2[:, 2] < self.DEPTH_THRESHOLD)
        else:
            near_mask = np.ones(pc1.shape[0], dtype=np.bool)
        indices = np.where(near_mask)[0]
        if len(indices) == 0:
            print('indices = np.where(mask)[0], len(indices) == 0')
            return None, None, None

        if self.num_points > 0:
            try:
                sampled_indices1 = np.random.choice(indices, size=self.num_points, replace=False, p=None)
                if self.no_corr:
                    sampled_indices2 = np.random.choice(indices, size=self.num_points, replace=False, p=None)
                else:
                    sampled_indices2 = sampled_indices1
            except ValueError:
                '''
                if not self.allow_less_points:
                    print('Cannot sample {} points'.format(self.num_points))
                    return None, None, None
                else:
                    sampled_indices1 = indices
                    sampled_indices2 = indices
                '''
                if not self.allow_less_points:
                    #replicate some points
                    sampled_indices1 = np.random.choice(indices, size=self.num_points, replace=True, p=None)
                    if self.no_corr:
                        sampled_indices2 = np.random.choice(indices, size=self.num_points, replace=True, p=None)
                    else:
                        sampled_indices2 = sampled_indices1
                else:
                    sampled_indices1 = indices
                    sampled_indices2 = indices
        else:
            sampled_indices1 = indices
            sampled_indices2 = indices

        pc1 = pc1[sampled_indices1]
        sf = sf[sampled_indices1]
        pc2 = pc2[sampled_indices2]

        return pc1, pc2, sf

    def __repr__(self):
        format_string = self.__class__.__name__ + '\n(data_process_args: \n'
        format_string += '\tDEPTH_THRESHOLD: {}\n'.format(self.DEPTH_THRESHOLD)
        format_string += '\tNO_CORR: {}\n'.format(self.no_corr)
        format_string += '\tallow_less_points: {}\n'.format(self.allow_less_points)
        format_string += '\tnum_points: {}\n'.format(self.num_points)
        format_string += ')'
        return format_string

class KITTI_hplflownet(torch.utils.data.Dataset): # hplflownet
    """
    Args:
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable):
        gen_func (callable):
        args:
    """

    def __init__(self,
                 transform=None,
                 train=False,
                 num_points=8192,
                 data_root='datasets/datasets_KITTI_hplflownet/',
                 remove_ground = True):
        self.root = osp.join(data_root, 'KITTI_processed_occ_final')
        #assert train is False
        self.train = train
        self.transform = ProcessData(data_process_args = {'DEPTH_THRESHOLD':35., 'NO_CORR':True},
                                    num_points=num_points,
                                    allow_less_points=False)
        self.num_points = num_points
        self.remove_ground = remove_ground

        self.samples = self.make_dataset()
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data_dict = {'index': index}
        pc1_loaded, pc2_loaded = self.pc_loader(self.samples[index])
        pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded])
        if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)
        pc1_transformed, pc2_transformed = torch.from_numpy(pc1_transformed).float(), torch.from_numpy(pc2_transformed).float()
        flow_3d = torch.from_numpy(sf_transformed).float() # .permute(1, 0)
        return pc1_transformed, pc2_transformed, flow_3d

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is removing ground: {}\n'.format(self.remove_ground)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str

    def make_dataset(self):
        do_mapping = True
        root = osp.realpath(osp.expanduser(self.root))

        all_paths = sorted(os.walk(root))
        useful_paths = [item[0] for item in all_paths if len(item[1]) == 0]
        try:
            assert (len(useful_paths) == 200)
        except AssertionError:
            print('assert (len(useful_paths) == 200) failed!', len(useful_paths))

        if do_mapping:
            mapping_path = osp.join(osp.dirname(__file__), 'KITTI_mapping.txt')
            print('mapping_path', mapping_path)

            with open(mapping_path) as fd:
                lines = fd.readlines()
                lines = [line.strip() for line in lines]
            useful_paths = [path for path in useful_paths if lines[int(osp.split(path)[-1])] != '']

        res_paths = useful_paths

        return res_paths

    def pc_loader(self, path):
        """
        Args:
            path:
        Returns:
            pc1: ndarray (N, 3) np.float32
            pc2: ndarray (N, 3) np.float32
        """
        pc1 = np.load(osp.join(path, 'pc1.npy'))  #.astype(np.float32)
        pc2 = np.load(osp.join(path, 'pc2.npy'))  #.astype(np.float32)

        if self.remove_ground:
            is_ground = np.logical_and(pc1[:,1] < -1.4, pc2[:,1] < -1.4)
            not_ground = np.logical_not(is_ground)

            pc1 = pc1[not_ground]
            pc2 = pc2[not_ground]

        return pc1, pc2

class KITTI_flownet3d(torch.utils.data.Dataset): # flownet3d
    def __init__(self, split='training150', root='datasets/KITTI_stereo2015', num_points = 8192):
        # assert os.path.isdir(root)
        assert split in ['training200', 'training160', 'training150', 'training40']

        self.root_dir = os.path.join(root, 'training')
        self.split = split
        self.augmentation = False
        self.n_points = num_points
        self.max_depth = 30

        if self.split == 'training200':
            self.indices = np.arange(200)
        if self.split == 'training150':
            self.indices = np.arange(150)
        elif self.split == 'training160':
            self.indices = [i for i in range(200) if i % 5 != 0]
        elif self.split == 'training40':
            self.indices = [i for i in range(200) if i % 5 == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if not self.augmentation:
            np.random.seed(23333)

        index = self.indices[i]
        path = os.path.join('./sceneflow_eval_dataset/kitti_rm_ground_KITTI_o', '%06d.npz' % index)
        data = np.load(path)
        pc1 = np.concatenate((data['pos1'][:,1:2], data['pos1'][:,2:3], data['pos1'][:,0:1]), axis=1)
        pc2 = np.concatenate((data['pos2'][:,1:2], data['pos2'][:,2:3], data['pos2'][:,0:1]), axis=1)
        flow_3d = np.concatenate((data['gt'][:,1:2], data['gt'][:,2:3], data['gt'][:,0:1]), axis=1)
        # limit max depth
        # pc1 = pc1[pc1[..., -1] < self.max_depth]
        # pc2 = pc2[pc2[..., -1] < self.max_depth]
        # flow_3d = flow_3d[pc2[..., -1] < self.max_depth]

        # random sampling
        indices1 = np.random.choice(pc1.shape[0], size=self.n_points, replace=pc1.shape[0] < self.n_points)
        indices2 = np.random.choice(pc2.shape[0], size=self.n_points, replace=pc2.shape[0] < self.n_points)
        pc1, pc2, flow_3d = pc1[indices1], pc2[indices2], flow_3d[indices1]

        return pc1, pc2, flow_3d


class KittiPointCloudDataset(Dataset):
    def __init__(self, path: str, max_dist: float = 40.0, num_points: int = 16384):
        self.path = path
        self.max_dist = max_dist
        self.num_points = num_points
        self.bin_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.bin')]

    def __len__(self):
        return len(self.bin_files)

    def __getitem__(self, idx):
        bin_file = self.bin_files[idx]
        pc = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
        pc = pc[:, :3]  # Ignore intensity
        pc = pc[(np.abs(pc) < self.max_dist).all(axis=1)]  # Filter points beyond max distance in all axes
        # if pc.shape[0] >= self.num_points:
        choice = np.random.choice(pc.shape[0], pc.shape[0], replace=False)
        pc = pc[choice, :]
        is_ground = pc[:,2] < -1.39
        not_ground = np.logical_not(is_ground)
        pc = pc[not_ground]

        pc2_path = './DATASETS/Driving_datasets/odKITTI/Raw_od_LIDAR/dataset/sequences/02/velodyne/000792.bin'
        pc2 = np.fromfile(pc2_path, dtype=np.float32).reshape(-1, 4)
        pc2 = pc2[:, :3]
        pc2 = pc2[(np.abs(pc2) < 40).all(axis=1)]
        choice = np.random.choice(pc2.shape[0], pc2.shape[0], replace=False)
        pc2 = pc2[choice, :]
        is_ground = pc2[:,2] < -1.39
        not_ground = np.logical_not(is_ground)
        pc2 = pc2[not_ground]
        return self.remove_outlier(pc), self.remove_outlier(pc2)
    
    def remove_outlier(self, pc, nb_neighbors=16, std_ratio=2.5):
        pcl = open3d.geometry.PointCloud()
        pcl.points = open3d.utility.Vector3dVector(pc)
        cl, ind = pcl.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        cl_arr = np.asarray(cl.points)
        return cl_arr
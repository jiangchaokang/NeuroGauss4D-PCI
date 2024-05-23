import os
import subprocess
from multiprocessing import Pool, Value
import tempfile
import argparse
from tqdm import tqdm
import time

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout.decode('utf-8'), stderr.decode('utf-8')

def process_scene(scene, gpu_id, dataset, dataset_path, iters, lr, layer_width, n_gaussians, counter):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write(scene)
        temp_file_path = temp_file.name
    
    os.chdir('./NeuroGauss4D')
    
    command = f"CUDA_VISIBLE_DEVICES={gpu_id} python main.py --dataset {dataset} --dataset_path {dataset_path} --scenes_list {temp_file_path} --num_points 8192 --interval 4 --iters {iters} --lr {lr} --layer_width {layer_width} --act_fn LeakyReLU --n_gaussians {n_gaussians} --scheduler poly"
    stdout, stderr = run_command(command)
    
    os.unlink(temp_file_path)
    
    with counter.get_lock():
        counter.value += 1
    
    return f"{scene}\n{stdout}\n{stderr}\n"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process scenes with NeuralPCI')
    parser.add_argument('--dataset', type=str, default='NL_Drive', help='Dataset name')
    parser.add_argument('--dataset_path', type=str, default='./data/NL_Drive/NL-Drive/test/', help='Dataset path')
    parser.add_argument('--iters', type=int, default=7000, help='Number of iterations')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--layer_width', type=int, default=800, help='Layer width')
    parser.add_argument('--n_gaussians', type=int, default=16, help='Number of Gaussians')
    parser.add_argument('--num_gpus', type=int, default=8, help='Number of GPUs')
    parser.add_argument('--max_processes_per_gpu', type=int, default=3, help='Maximum processes per GPU')
    parser.add_argument('--output', type=str, default='./NeuroGauss4D/log/4DGS_Exper_macpool_v3/NL_test0.txt', help='Output file path')
    parser.add_argument('--scenes_list_file', type=str, default='./NeuroGauss4D/data/list/NL_Drive_test_1.txt', help='Scenes list file path')
    args = parser.parse_args()
    
    with open(args.scenes_list_file, 'r') as f:
        scenes = f.read().splitlines()
    
    pool = Pool(processes=args.num_gpus * args.max_processes_per_gpu)
    
    counter = Value('i', 0)
    results = []
    
    for i, scene in enumerate(scenes):
        gpu_id = i % args.num_gpus
        result = pool.apply_async(process_scene, args=(scene, gpu_id, args.dataset, args.dataset_path, args.iters, args.lr, args.layer_width, args.n_gaussians, counter))
        results.append(result)
    
    with tqdm(total=len(scenes), desc='Processing scenes') as pbar:
        while counter.value < len(scenes):
            pbar.n = counter.value
            pbar.refresh()
            time.sleep(1)
    
    pool.close()
    pool.join()
    
    output_lines = []
    for result in results:
        output_lines.append(result.get())
    
    with open(args.output, 'w') as f:
        f.writelines(output_lines)

###  python ./NeuroGauss4D/script/run/dist_infer_NL.py --output ./NeuroGauss4D/log/4DGS_Exper_macpool_v3/NL_test1.txt --scenes_list_file ./NeuroGauss4D/data/list/NL_Drive_test_1.txt


# python ./NeuroGauss4D/script/run/dist_infer_NL.py --output ./NeuroGauss4D/log/4DGS_Exper_macpool_v3/NL_test2.txt --scenes_list_file ./NeuroGauss4D/data/list/NL_Drive_test_2.txt
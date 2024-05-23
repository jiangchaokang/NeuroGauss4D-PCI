import os
import subprocess
from multiprocessing import Pool
import tempfile
import argparse
import math
def write_result_to_file(result):
    with open(args.output, 'a') as output_file:
        output_file.write(result + '\n')

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout.decode('utf-8'), stderr.decode('utf-8')

def process_scenes(scenes, gpu_id, dataset, dataset_path, iters, lr, layer_width, n_gaussians):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write('\n'.join(scenes))
        temp_file_path = temp_file.name
    
    os.chdir('./NeuroGauss4D')
    # --Att_Fusion --NeuralField --T_RBF_GMM --Gaussians4D --Gaussian_Rep
    command = f"CUDA_VISIBLE_DEVICES={gpu_id} python main.py --dataset {dataset} --dataset_path {dataset_path} --scenes_list {temp_file_path} --num_points 8192 --interval 4 --iters {iters} --lr {lr} --layer_width {layer_width} --act_fn LeakyReLU --n_gaussians {n_gaussians} --scheduler poly --Gaussian_Rep"
    stdout, stderr = run_command(command)
    
    os.unlink(temp_file_path)
    
    return f"{''.join(scenes)}\n{stdout}\n{stderr}\n"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process scenes with NeuralPCI')
    parser.add_argument('--dataset', type=str, default='NL_Drive', help='Dataset name')
    parser.add_argument('--dataset_path', type=str, default='./data/NL_Drive/NL-Drive/test/', help='Dataset path')
    parser.add_argument('--iters', type=int, default=6000, help='Number of iterations')
    parser.add_argument('--lr', type=float, default=0.009, help='Learning rate')
    parser.add_argument('--layer_width', type=int, default=1280, help='Layer width')
    parser.add_argument('--n_gaussians', type=int, default=16, help='Number of Gaussians')
    parser.add_argument('--num_gpus', type=int, default=8, help='Number of GPUs')
    parser.add_argument('--max_processes_per_gpu', type=int, default=4, help='Maximum processes per GPU')
    parser.add_argument('--output', type=str, default='./NeuroGauss4D/log/Ablation_Experiment/B_NL_test2.txt', help='Output file path')
    parser.add_argument('--scenes_list_file', type=str, default='./NeuroGauss4D/data/list/NL_Drive_test_2.txt', help='Scenes list file path')
    args = parser.parse_args()
    
    with open(args.scenes_list_file, 'r') as f:
        scenes = f.read().splitlines()
    
    num_processes = args.num_gpus * args.max_processes_per_gpu
    scenes_per_process = math.ceil(len(scenes) / num_processes)
    
    pool = Pool(processes=num_processes)
    
    # results = []
    # for i in range(num_processes):
    #     start_index = i * scenes_per_process
    #     end_index = min((i + 1) * scenes_per_process, len(scenes))
    #     gpu_id = i % args.num_gpus
    #     process_scenes_list = scenes[start_index:end_index]
    #     # print('start_index:', start_index, "end_index:", end_index, '\n', process_scenes_list, '\n' "gpu_id:", gpu_id)

    #     result = pool.apply_async(process_scenes, args=(process_scenes_list, gpu_id, args.dataset, args.dataset_path, args.iters, args.lr, args.layer_width, args.n_gaussians))
    #     results.append(result)
    #     print("number results:", len(results))
    
    # pool.close()
    # pool.join()
    
    # output_lines = []
    # for result in results:
    #     output_lines.append(result.get())
    
    # with open(args.output, 'w') as f:
    #     f.writelines(output_lines)

    for i in range(num_processes):
        start_index = i * scenes_per_process
        end_index = min((i + 1) * scenes_per_process, len(scenes))
        gpu_id = i % args.num_gpus
        process_scenes_list = scenes[start_index:end_index]

        pool.apply_async(
            process_scenes, 
            args=(process_scenes_list, gpu_id, args.dataset, args.dataset_path, args.iters, args.lr, args.layer_width, args.n_gaussians),
            callback=write_result_to_file 
        )

    pool.close()
    pool.join()


###  python ./NeuroGauss4D/script/run/dist_infer_NL.py --output ./NeuroGauss4D/log/4DGS_Exper_macpool_v3/wo_mlp/NL_test1.txt --scenes_list_file ./NeuroGauss4D/data/list/NL_Drive_test_1.txt --layer_width 32 
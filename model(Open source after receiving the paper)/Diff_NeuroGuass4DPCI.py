
######################################## ************************ ########################################
"""
This code is a draft and not related to the main text.
The complete code will be released after the paper is accepted.
"""
######################################## ************************ ########################################
import torch
import torch.nn as nn
import numpy as np
import pdb

import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

##  pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple torch_geometric egnn_pytorch


class GaussianMixtureModel(nn.Module):
    def __init__(self, in_channels, n_kernels):
        super().__init__()
        self.n_kernels = n_kernels
        self.mu = nn.Parameter(torch.randn(n_kernels, in_channels) / np.sqrt(n_kernels))
        self.sigma = nn.Parameter(torch.ones(n_kernels, in_channels))
        self.p = nn.Parameter(torch.ones(n_kernels) / n_kernels)

    def forward(self, x):
        x = x.unsqueeze(1) - self.mu.unsqueeze(0)
        z = torch.sum(x**2 / (self.sigma.unsqueeze(0)**2), dim=-1)
        p = torch.softmax(torch.log(self.p) - 0.5 * z, dim=-1)
        mu_x = torch.sum(p.unsqueeze(-1) * self.mu.unsqueeze(0), dim=1)
        return mu_x


# class TemporalAttention(nn.Module):
#     def __init__(self, layer_width, num_heads):
#         super(TemporalAttention, self).__init__()
#         self.layer_width = layer_width
#         self.num_heads = num_heads
#         self.head_dim = layer_width // num_heads
        
#         self.query = nn.Linear(layer_width, layer_width)
#         self.key = nn.Linear(layer_width, layer_width)
#         self.value = nn.Linear(layer_width, layer_width)
        
#         self.out = nn.Linear(layer_width, layer_width)
    
#     def forward(self, x):
#         seq_len, _ = x.size()
        
#         q = self.query(x).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
#         k = self.key(x).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
#         v = self.value(x).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        
#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
#         attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        
#         attn_output = torch.matmul(attn_probs, v).transpose(0, 1).contiguous().view(seq_len, self.layer_width)
        
#         return self.out(attn_output)
    
class TemporalAttention(nn.Module):
    def __init__(self, layer_width, num_heads):
        super(TemporalAttention, self).__init__()
        self.layer_width = layer_width
        self.num_heads = num_heads
        self.bottleneck_size = layer_width//4
        self.head_dim = self.bottleneck_size // num_heads

        self.shared_linear = nn.Linear(layer_width, self.bottleneck_size)
        
        self.out = nn.Linear(self.bottleneck_size, layer_width)
    
    def forward(self, x):
        seq_len, _ = x.size()
        
        shared_qkv = self.shared_linear(x)
        q = shared_qkv.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        k = shared_qkv.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        v = shared_qkv.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        attn_output = torch.matmul(attn_probs, v).transpose(0, 1).contiguous().view(seq_len, self.bottleneck_size)
        
        output = self.out(attn_output)
        
        return output

class ProbabilisticGraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(ProbabilisticGraphLayer, self).__init__(aggr='add') 
        self.lin = nn.Linear(in_channels, out_channels)
        self.score_net = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        x = self.lin(x)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j, edge_index, size):
        score = self.score_net(torch.cat([x_i, x_j], dim=-1))
        return score * x_j

class ProbabilisticGraphNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ProbabilisticGraphNetwork, self).__init__()
        self.conv1 = ProbabilisticGraphLayer(in_channels, hidden_channels)
        self.conv2 = ProbabilisticGraphLayer(hidden_channels, hidden_channels)
        self.conv3 = ProbabilisticGraphLayer(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x


class NeuroGauss4DPCI(torch.nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        dim_pc = args.dim_pc
        dim_time = args.dim_time
        layer_width = args.layer_width 
        act_fn = args.act_fn
        norm = args.norm
        depth_encode = args.depth_encode
        depth_pred = args.depth_pred
        pe_mul = args.pe_mul
        self.n_gaussians = args.n_gaussians
        self.gmm = GaussianMixtureModel(args.layer_width, self.n_gaussians)
        self.sigma = args.sigma

        # self.temporal_attention = TemporalAttention(layer_width, num_heads=4)
        self.ProbabilisticGraphNet = ProbabilisticGraphNetwork(layer_width, layer_width//4, layer_width)

        '''Point cloud density: For dense point clouds, smaller sigma values can be used because the distance between points is smaller and does not require excessive smoothing. For sparse point clouds, larger sigma values can be used to fill the gaps between points.
            Noise level: If the point cloud contains a lot of noise, using a larger sigma value can effectively suppress the noise and improve the smoothing effect. If the point cloud noise is small, a smaller sigma value can be used to preserve more details.
            Required smoothness level: Depending on the specific application, the sigma value can be adjusted to obtain the desired smoothness level. A larger sigma value will produce smoother results, but may result in loss of some details; A smaller sigma value will retain more details, but may not be smooth enough.
            As a preliminary suggestion, you can try the following sigma values:
            For dense point clouds, a sigma value between 0.01 and 0.05 can be attempted.
            For sparse point clouds, a sigma value between 0.05 and 0.1 can be attempted.
            If the point cloud contains a lot of noise, the sigma value can be appropriately increased
        '''
        if args.use_rrf:
            dim_rrf = args.dim_rrf
            self.transform = 0.1 * torch.normal(0, 1, size=[dim_pc, dim_rrf]).cuda()
        else:
            dim_rrf = dim_pc

        # input layer
        self.layer_input = NeuralPCI_Layer(dim_in = (dim_rrf + dim_time) * pe_mul, 
                                           dim_out = layer_width, 
                                           norm = norm,
                                           act_fn = act_fn
                                           )

        # hidden layers
        self.hidden_encode = NeuralPCI_Block(depth = depth_encode, 
                                             width = layer_width, 
                                             norm = norm,
                                             act_fn = act_fn
                                             )

        # insert interpolation time
        self.layer_time = NeuralPCI_Layer(dim_in = layer_width + dim_time * pe_mul, 
                                          dim_out = layer_width, 
                                          norm = norm,
                                          act_fn = act_fn
                                          )

        # hidden layers
        self.hidden_pred = NeuralPCI_Block(depth = depth_pred, 
                                           width = layer_width, 
                                           norm = norm,
                                           act_fn = act_fn
                                           )

        # output layer
        self.layer_output = NeuralPCI_Layer(dim_in = layer_width, 
                                          dim_out = dim_pc, 
                                          norm = norm,
                                          act_fn = None
                                          )
        
        # zero init for last layer
        if args.zero_init:
            for m in self.layer_output.layer:
                if isinstance(m, nn.Linear):
                    # torch.nn.init.normal_(m.weight.data, 0, 0.01)
                    m.weight.data.zero_()
                    m.bias.data.zero_()
    
        self.num_timesteps = args.num_timesteps
        self.beta_start = 1e-4 # args.beta_start
        self.beta_end = 0.02 # args.beta_end
        self.betas = self.get_beta_schedule().cuda()
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1).to(self.alphas_cumprod.device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def get_beta_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
    
    def posenc(self, x):
        sinx = torch.sin(x)
        cosx = torch.cos(x)
        x = torch.cat((x, sinx, cosx), dim=1)
        return x
    
    def q_sample(self, pc_start, t, noise_factor=0.1):
        noise = torch.randn_like(pc_start) * noise_factor
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alphas_cumprod_t * pc_start + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def p_sample(self, pc_t, t, time_pred, time_current):
        betas_t = self.betas[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).view(-1, 1)
        
        model_mean = sqrt_recip_alphas_t * (pc_t - betas_t * self.model(pc_t, time_current, time_pred) / sqrt_one_minus_alphas_cumprod_t)
        
        posterior_variance_t = self.posterior_variance[t].view(-1, 1)
        noise = torch.randn_like(pc_t)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

    def forward(self, pc_current, time_current, time_pred, train=True, num_denoise_steps=3):
        # Convert time_current and time_pred to PyTorch tensors with the same data type and device as pc_current
        time_current = torch.tensor(time_current).repeat(pc_current.shape[0], 1).to(pc_current.device).float().detach()
        time_pred = torch.tensor(time_pred).repeat(pc_current.shape[0], 1).to(pc_current.device).float().detach()
        
        if train:
            t = torch.randint(0, self.num_timesteps, (1,)).to(pc_current.device)
            pc_noisy, _ = self.q_sample(pc_current, t)
            pc_recon = self.model(pc_noisy, time_current, time_pred)
            return pc_recon
        else:
            for i in range(num_denoise_steps):
                t = torch.tensor([i]).to(pc_current.device)
                pc_current = self.p_sample(pc_current, t, time_pred, time_current)
            pc_pred = self.model(pc_current, time_current, time_pred)
            return pc_pred
        
    def model(self, pc_current, t, time_pred):
        if self.args.use_rrf:
            pc_current = torch.matmul(2. * torch.pi * pc_current, self.transform)
        if t.shape[0] != pc_current.shape[0]:
            time_t = t.view(-1, 1).repeat(1, pc_current.shape[0]).view(-1, 1)
        else:
            time_t = t
        x = torch.cat((pc_current, time_t), dim=1)
        x = self.posenc(x)
        x = self.layer_input(x)

        x = self.hidden_encode(x)
        # pdb.set_trace()
        data = Data(x=x, pos=pc_current)
        edge_index = knn_graph(data.pos, k=16, loop=False)
        data.edge_index = edge_index
        x = self.ProbabilisticGraphNet(data)
        
        time_pred = self.posenc(time_pred)
        x = torch.cat((x, time_pred), dim=1)
        x = self.layer_time(x)
        # x = self.temporal_attention(x).squeeze()

        x = self.hidden_pred(x)
        x = self.gmm(x)

        x = self.layer_output(x)

        # return x
        pc_pred = x

        pc_pred_min = pc_pred.min(dim=0, keepdim=True)[0]
        pc_pred_max = pc_pred.max(dim=0, keepdim=True)[0]
        pc_pred_normalized = (pc_pred - pc_pred_min) / (pc_pred_max - pc_pred_min + 1e-6)

        kernel_size = int(self.sigma * 6)
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        gaussian_kernel = self._gaussian_kernel(kernel_size, self.sigma).cuda()

        pc_pred_normalized = pc_pred_normalized.transpose(0, 1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        pc_pred_splatted = nn.functional.conv3d(pc_pred_normalized, gaussian_kernel, padding=kernel_size//2)
        pc_pred_splatted = pc_pred_splatted.squeeze(0).squeeze(0).squeeze(0).transpose(0, 1)

        pc_pred = pc_pred_splatted * (pc_pred_max - pc_pred_min + 1e-6) + pc_pred_min

        
        return pc_pred

    def _gaussian_kernel(self, kernel_size, sigma):
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size, kernel_size).view(kernel_size, kernel_size, kernel_size)
        y_grid = x_grid.transpose(1, 0)
        z_grid = x_grid.transpose(2, 0)
        xyz_grid = torch.stack([x_grid, y_grid, z_grid], dim=-1).float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        gaussian_kernel = (1./(2.*np.pi*variance)**(3/2)) * torch.exp(-torch.sum((xyz_grid - mean)**2., dim=-1) / (2*variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        return gaussian_kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
import functools
import math
import os
import time
from tkinter import W
from typing import Callable, Optional, Tuple
from tqdm import tqdm
import nerfacc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from torch_scatter import scatter_add

parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
        name='render_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
        verbose=True)

total_variation_cuda = load(
        name='total_variation_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/total_variation.cpp', 'cuda/total_variation_kernel.cu']],
        verbose=True)

class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_views=3, input_ch_time=9, skips=[],):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_time = input_ch_time
        self.skips = skips
        self._time, self._time_out = self.create_net()

    def create_net(self):
        layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 2):
            layer = nn.Linear
            in_channels = self.W
            if i in self.skips:
                in_channels += self.input_ch
            layers += [layer(in_channels, self.W)]
        return nn.ModuleList(layers), nn.Linear(self.W, 3)

    def query_time(self, new_pts, t, net, net_final):
        h = torch.cat([new_pts, t], dim=-1)
        for i, l in enumerate(net):
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([new_pts, h], -1)
        return net_final(h)

    def forward(self, input_pts, ts):
        dx = self.query_time(input_pts, ts, self._time, self._time_out)
        input_pts_orig = input_pts[:, :3]
        out=input_pts_orig + dx
        return out

# Model
class RGBNet(nn.Module):
    def __init__(self, D=3, W=256, h_ch=256, views_ch=33, pts_ch=27, times_ch=17, output_ch=3):
        """ 
        """
        super(RGBNet, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = h_ch
        self.input_ch_views = views_ch
        self.input_ch_pts = pts_ch
        self.input_ch_times = times_ch
        self.output_ch = output_ch
        self.feature_linears = nn.Linear(self.input_ch, W)
        self.views_linears = nn.Sequential(nn.Linear(W+self.input_ch_views, W//2),nn.ReLU(),nn.Linear(W//2, self.output_ch))
        
    def forward(self, input_h, input_views):
        feature = self.feature_linears(input_h)
        feature_views = torch.cat([feature, input_views],dim=-1)
        outputs = self.views_linears(feature_views)
        return outputs

'''Model'''
class DAPNeRF(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0, add_cam=False,
                 alpha_init=None, fast_color_thres=0,
                 voxel_dim=0, defor_depth=3, net_width=128,
                 posbase_pe=10, viewbase_pe=4, timebase_pe=8, gridbase_pe=2,
                 # for nerfacc occgrid
                 occ_grid_reso=-1, occ_alpha_thres=0.0, occ_thres=0.01, occ_ema_decay=0.95,
                 # for particle
                 particle_num=1000, enabled_particle=False,
                 **kwargs):
        
        super(DAPNeRF, self).__init__()
        self.add_cam = add_cam
        self.voxel_dim = voxel_dim
        self.defor_depth = defor_depth
        self.net_width = net_width
        self.posbase_pe = posbase_pe
        self.viewbase_pe = viewbase_pe
        self.timebase_pe = timebase_pe
        self.gridbase_pe = gridbase_pe
        times_ch = 2*timebase_pe+1
        views_ch = 3+3*viewbase_pe*2
        pts_ch = 3+3*posbase_pe*2,
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres
        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.act_shift = np.log(1/(1-alpha_init) - 1)
        print('DAPNeRF: set density bias shift to', self.act_shift)
        
        self.occ_grid_reso = occ_grid_reso
        self.occ_alpha_thres = occ_alpha_thres
        self.occ_thres = occ_thres
        self.occ_ema_decay = occ_ema_decay
        self.use_occ_grid = self.occ_grid_reso > 0
        self.occ_grid = None
        if self.use_occ_grid:
            self.occ_grid = nerfacc.OccGridEstimator(
                roi_aabb=torch.cat([self.xyz_min, self.xyz_max]),
                resolution=self.occ_grid_reso,
            )

            def occ_eval_fn(x, step_size):
                density, _ = self.query_density(
                    x, torch.zeros_like(x[:, :1]) + torch.randn(1, 1))
                return density * step_size

            self.occ_eval_fn = occ_eval_fn

        timenet_width = net_width
        timenet_depth = 1
        timenet_output = voxel_dim+voxel_dim*2*gridbase_pe
        self.timenet = nn.Sequential(
            nn.Linear(times_ch, timenet_width), nn.ReLU(inplace=True),
            nn.Linear(timenet_width, timenet_output)
        )
        if self.add_cam == True:
            views_ch = 3+3*viewbase_pe*2+timenet_output
            self.camnet = nn.Sequential(
            nn.Linear(times_ch, timenet_width), nn.ReLU(inplace=True),
            nn.Linear(timenet_width, timenet_output))
            print('DAPNeRF: camnet', self.camnet)

        featurenet_width = net_width
        featurenet_depth = 1
        grid_dim = voxel_dim+voxel_dim*2*gridbase_pe  # voxel_dim*3+voxel_dim*3*2*gridbase_pe
        input_dim = grid_dim+0+0+3+3*posbase_pe*2  # +timenet_output
        self.featurenet = nn.Sequential(
            nn.Linear(input_dim, featurenet_width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(featurenet_width, featurenet_width), nn.ReLU(inplace=True))
                for _ in range(featurenet_depth-1)
            ],
            )
        self.featurenet_width = featurenet_width
        self._set_grid_resolution(num_voxels)
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=3+3*posbase_pe*2, input_ch_time=timenet_output)
        input_dim = featurenet_width
        self.densitynet = nn.Linear(input_dim, 1)

        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('grid_poc', torch.FloatTensor([(2**i) for i in range(gridbase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('view_poc', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))

        self.voxel_dim = voxel_dim
        self.feature= torch.nn.Parameter(torch.zeros([1, self.voxel_dim, *self.world_size], dtype=torch.float32))
        self.rgbnet = RGBNet(W=net_width, h_ch=featurenet_width, views_ch=views_ch, pts_ch=pts_ch, times_ch=times_ch)

        print('DAPNeRF: feature voxel grid', self.feature.shape)
        print('DAPNeRF: timenet mlp', self.timenet)
        print('DAPNeRF: deformation_net mlp', self.deformation_net)
        print('DAPNeRF: densitynet mlp', self.densitynet)
        print('DAPNeRF: featurenet mlp', self.featurenet)
        print('DAPNeRF: rgbnet mlp', self.rgbnet)

        # added particle
        self.particle_num = particle_num
        self.particle_pts_0 = torch.nn.Parameter(torch.randn(particle_num, 3))
        self.particle_feat = torch.nn.Parameter(
            torch.zeros(particle_num, self.voxel_dim))  # Compatible with feature grid
        self.enabled_particle = enabled_particle  # if already enable

        # additional
        self.dynamic_static_only = None
        self.particle_pts_t = None

        # hyper-nerf data
        self.bg_mask_grid = None
  

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()

        # [move from forward] pad to enable "mult_dist_interp", in advance
        x_pad = math.ceil((self.world_size[0]-1)/4.0)*4-self.world_size[0]+1
        y_pad = math.ceil((self.world_size[1]-1)/4.0)*4-self.world_size[1]+1
        z_pad = math.ceil((self.world_size[2]-1)/4.0)*4-self.world_size[2]+1
        self.world_size += torch.tensor([x_pad, y_pad, z_pad], dtype=torch.long)

        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('DAPNeRF: voxel_size      ', self.voxel_size)
        print('DAPNeRF: world_size      ', self.world_size)
        print('DAPNeRF: voxel_size_base ', self.voxel_size_base)
        print('DAPNeRF: voxel_size_ratio', self.voxel_size_ratio)


    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'act_shift': self.act_shift,
            'voxel_size_ratio': self.voxel_size_ratio,
            'fast_color_thres': self.fast_color_thres,
            'voxel_dim':self.voxel_dim,
            'defor_depth':self.defor_depth,
            'net_width':self.net_width,
            'posbase_pe':self.posbase_pe,
            'viewbase_pe':self.viewbase_pe,
            'timebase_pe':self.timebase_pe,
            'gridbase_pe':self.gridbase_pe,
            'add_cam': self.add_cam,
            "occ_grid_reso": self.occ_grid_reso,
            "occ_alpha_thres": self.occ_alpha_thres,
            "occ_thres": self.occ_thres,
            "occ_ema_decay": self.occ_ema_decay,
            # particle
            "enabled_particle": self.enabled_particle,
            "particle_num": self.particle_num,
        }


    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('DAPNeRF: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('DAPNeRF: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)
        self.feature = torch.nn.Parameter(
            F.interpolate(self.feature.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
 
    def feature_total_variation_add_grad(self, weight, dense_mode):
        weight = weight * self.world_size.max() / 128
        total_variation_cuda.total_variation_add_grad(
            self.feature.float(), self.feature.grad.float(), weight, weight, weight, dense_mode)

    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True):
        '''Wrapper for the interp operation'''
        mode = 'bilinear'
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        ret_lst = [
            F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1])
            for grid in grids
        ]
        for i in range(len(grids)):
            if ret_lst[i].shape[-1] == 1:
                ret_lst[i] = ret_lst[i].squeeze(-1)
        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst

    def init_particles(self, bg_points=None):
        self.enabled_particle = True
        
        # use learned occ mask (level=1) to sample particles 
        occ_grid = self.occ_grid
        binaries = occ_grid.binaries[0]

        # [optional for hyper-nerf data] exclude bg
        if bg_points is not None:
            bg_points_one = torch.ones(bg_points.shape[0], 1)
            bg_mask_grid = torch.zeros_like(binaries, dtype=torch.float)[None, None, ...]
            bg_mask_grid = self.particle_to_grid(bg_points, bg_points_one, bg_mask_grid)
            self.bg_mask_grid = bg_mask_grid>0.

            particle_init_mask = binaries & (~self.bg_mask_grid)
        else:
            particle_init_mask = binaries

        # grid_coords-ijk
        grid_coords = occ_grid.grid_coords
        non_empty_grid_coords = grid_coords[particle_init_mask.flatten()]

        # sample particles arrangement
        N_node = non_empty_grid_coords.shape[0]
        N_p_each_node = self.particle_num // N_node
        N_p_rest = self.particle_num - N_p_each_node*N_node
        
        # pre allocate space
        particles = torch.zeros(self.particle_num, 3, device=self.particle_feat.device)
        if N_p_each_node > 0:
            x = non_empty_grid_coords.view(N_node, 1, 3) + torch.rand(N_node, N_p_each_node, 3)
            x = x.view(-1, 3) / occ_grid.resolution  # grid coord(ijk) to voxel coord
            x = occ_grid.aabbs[0, :3] + x * (
                occ_grid.aabbs[0, 3:] - occ_grid.aabbs[0, :3]
            )  # voxel coordinates [0, 1]^3 -> world
            particles[:N_node*N_p_each_node] = x
        
        if N_p_rest > 0:
            selected_node = torch.randperm(N_node)[:N_p_rest]
            x = non_empty_grid_coords[selected_node, :] + torch.rand(N_p_rest, 3)
            x = x / occ_grid.resolution  # grid coord(ijk) to voxel coord
            x = occ_grid.aabbs[0, :3] + x * (
                occ_grid.aabbs[0, 3:] - occ_grid.aabbs[0, :3]
            )  # voxel coordinates [0, 1]^3 -> world
            particles[N_node*N_p_each_node:] = x
        
        # Register
        self.particle_pts_0 = torch.nn.Parameter(particles)


    def particle_to_grid(self, particle_pts_t, particle_feat, grid, weight_norm=False):
        aabb_size = (self.xyz_max - self.xyz_min)
        particle_pts_t = torch.clamp(  # ensure within the bbox
                particle_pts_t, min=self.xyz_max-(aabb_size*0.99), max=self.xyz_min+(aabb_size*0.99))

        # coord translate
        device = particle_pts_t.device
        N = particle_pts_t.shape[0]
        W, H, D = grid.shape[-3:]
        xyz = particle_pts_t  # .reshape(1, 1, 1, -1, 3)
        vox_sz = (self.xyz_max - self.xyz_min) / torch.tensor([W-1, H-1, D-1])  # align corners
        particle_ijk = (xyz - self.xyz_min.view(1, 3)) / vox_sz.view(1, 3)
        base = torch.floor(particle_ijk)
        
        # interpolation weight
        base_shift = particle_ijk - base
        inter_weight = [1 - base_shift, base_shift]
        p_ngbr_weight = torch.zeros((N, 8), device=device)
        p_ngbr_ijk = torch.zeros((N, 8, 3), device=device, dtype=torch.long)
        # for 8 neighbors
        for add_i in range(2):
            for add_j in range(2):
                for add_k in range(2):
                    ngbr_id = add_i*4 + add_j*2 + add_k
                    ngbr_weight = inter_weight[add_i][:, 0] * inter_weight[add_j][:, 1] * inter_weight[add_k][:, 2]
                    p_ngbr_weight[:, ngbr_id] = ngbr_weight
                    p_ngbr_ijk[:, ngbr_id, :] = base + torch.tensor([add_i, add_j, add_k])

        # node index [N, 8]
        p_ngbr_ijk_flat = p_ngbr_ijk[:, :, 0]*H*D + p_ngbr_ijk[:, :, 1]*D + p_ngbr_ijk[:, :, 2]
        # feature contribution from p to ngbr [N, 8, C]
        p_ngbr_feat_contri = particle_feat.view(N, 1, -1) * p_ngbr_weight.view(N, 8, 1)

        # node summarize [n_node_grid, C]
        n_node_grid = W * H * D
        # assert (p_ngbr_ijk_flat < n_node_grid).all(), 'ijk error'
        p2g_res = scatter_add(p_ngbr_feat_contri.view(N*8, -1), p_ngbr_ijk_flat.view(N*8), dim=0, dim_size=n_node_grid)     
        if weight_norm:
            node_weight_sum = scatter_add(p_ngbr_weight.view(N*8, -1), p_ngbr_ijk_flat.view(N*8), dim=0, dim_size=n_node_grid)
            node_weight_sum[node_weight_sum==0] = 1.  # avoid zero-divide
            p2g_res /= node_weight_sum
        p2g_res = p2g_res.transpose(1, 0)  # [C, n_node_grid]

        # partially override static grid
        override_mask = torch.ones((1, n_node_grid), dtype=torch.bool, device=device)
        override_mask[:, p_ngbr_ijk_flat.view(N*8)] = False
        grid_flat = grid.view(-1, n_node_grid)
        grid_flat = grid_flat * override_mask + p2g_res

        # soft fuse
        # override_mask = torch.zeros((1, n_node_grid), dtype=torch.bool, device=device)
        # override_mask[:, p_ngbr_ijk_flat.view(N*8)] = True
        # dyn_grid_prob_flat = F.sigmoid(self.dyn_grid_pred_prob).view(-1, n_node_grid)
        # dyn_grid_prob_flat = dyn_grid_prob_flat * override_mask
        # grid_flat = grid.view(-1, n_node_grid)
        # grid_flat = grid_flat*(1-dyn_grid_prob_flat) + p2g_res*dyn_grid_prob_flat

        return grid_flat.view_as(grid)
    
    def save_t0_particle(self):  # for debug
        old_particle_num = self.particle_num
        particle_pts_t_0 = self.particle_deform_t(torch.zeros(1 ,1))
        particle_pts_t_0 = torch.clamp(  # ensure within the bbox
            particle_pts_t_0, min=self.xyz_min*0.99, max=self.xyz_max*0.99)

        time_info = torch.zeros(old_particle_num, 1)
        sigma_p_t, _ = self.query_density(particle_pts_t_0, time_info)
        alpha_p_t = 1 - torch.exp(-sigma_p_t * 0.5 * self.voxel_size)

        # occ
        occ_grid = self.occ_grid
        binaries = occ_grid.binaries[0]
        grid_coords = occ_grid.grid_coords
        non_empty_grid_coords = grid_coords[binaries.flatten()]
        
        torch.save(particle_pts_t_0, "particle_pts_t_0.pth")
        torch.save(sigma_p_t, "particle_sigma_t_0.pth")
        torch.save(alpha_p_t, "particle_alpha_t_0.pth")
        torch.save(non_empty_grid_coords, "non_empty_grid_coords.pth")
    
    def _grid_tv_loss(self, grid):
        # batch(1), C, Nx, Ny, Nz
        _, C, Nx, Ny, Nz = grid.shape

        # compute tv
        kernel_x = torch.tensor([[[[[1]], [[-1]]]]], dtype=torch.float32)
        kernel_y = torch.tensor([[[[[1], [-1]]]]], dtype=torch.float32)
        kernel_z = torch.tensor([[[[[1, -1]]]]], dtype=torch.float32)
        kernel_x = kernel_x.repeat(C, 1, 1, 1, 1)  # repeat channel
        kernel_y = kernel_y.repeat(C, 1, 1, 1, 1)  # weight with shape:
        kernel_z = kernel_z.repeat(C, 1, 1, 1, 1)  # [out_c, in_c/n_group, kx, ky, kz]

        diff_x = F.conv3d(grid, kernel_x, groups=C)
        diff_y = F.conv3d(grid, kernel_y, groups=C)
        diff_z = F.conv3d(grid, kernel_z, groups=C)

        # L2 dist
        diff_x = torch.norm(diff_x, dim=1)
        diff_y = torch.norm(diff_y, dim=1)
        diff_z = torch.norm(diff_z, dim=1)
        tv_loss = (diff_x.sum() + diff_y.sum() + diff_z.sum())/torch.count_nonzero(diff_x)

        return tv_loss


    def motion_tv_loss(self, t=0):
        option = 1
        if option == 1:  # as rigid as possible
            # particle_pts_t = self.particle_deform_t(torch.tensor([[t]], dtype=torch.float32))
            particle_pts_t = self.particle_pts_t
            dpts = particle_pts_t - self.particle_pts_0

        elif option ==2:  # dx/dt using torch.autograd.grad
            # t = torch.tensor([[t]], dtype=torch.float32, requires_grad=True)
            # jacobian(self.particle_deform_t, t, create_graph=True, vectorize=True)
            pass
        else:  # dx/dt using numerical subtraction
            particle_pts_t = self.particle_deform_t(torch.tensor([[t]], dtype=torch.float32))
            particle_pts_t1 = self.particle_deform_t(torch.tensor([[t+0.1]], dtype=torch.float32))
            dpts = particle_pts_t1 - particle_pts_t
        dpts_grid = torch.zeros([1, 3, *(self.world_size)], dtype=torch.float32)
        dpts_grid = self.particle_to_grid(particle_pts_t, dpts, dpts_grid, weight_norm=True)

        return self._grid_tv_loss(dpts_grid)
    
    def fuse_feature_tv_loss(self):
        return self._grid_tv_loss(self.fused_feature_grid)

    @torch.no_grad()
    def remove_redundant_particle(self, time_sel, traj_thres, step_interval, alpha_thres, strict_traj=True):
        device = self.feature.device
        old_particle_num = self.particle_num

        # occ_grid = self.occ_grid.binaries[0][None, None, ...].float()

        # temporary container
        p_alive_mark = torch.ones(old_particle_num, dtype=torch.bool, device=device)
        uninformative_mask = torch.zeros(old_particle_num, dtype=torch.bool, device=device)
        informative_mask = torch.zeros(old_particle_num, dtype=torch.bool, device=device)
        out_of_range_mask = torch.zeros(old_particle_num, dtype=torch.bool, device=device)
        in_bg_mask = torch.zeros(old_particle_num, dtype=torch.bool, device=device)

        particle_traj_len = torch.zeros(old_particle_num, device=device)
        particle_pts_t_last = None
        particle_pts_t_0 = None
        for t in range(time_sel.shape[0]):
            # compute particle at `t`
            particle_pts_t = self.particle_deform_t(time_sel[t].view(1, 1))
            aabb_size = (self.xyz_max - self.xyz_min)
            particle_pts_t = torch.clamp(  # ensure within the bbox
                    particle_pts_t, min=self.xyz_max-(aabb_size*0.99), max=self.xyz_min+(aabb_size*0.99))

            # accumulate traj
            if particle_pts_t_last is not None:  # t(0)
                particle_traj_len += torch.norm(particle_pts_t-particle_pts_t_last, dim=1)
            else:  # further usage
                particle_pts_t_0 = particle_pts_t
            particle_pts_t_last = particle_pts_t

            # uninformative particles
            time_info = time_sel[t:t+1].view(1, 1).expand(old_particle_num, 1)
            alpha_p_t, _ = self.query_density(particle_pts_t, time_info)
            alpha = 1 - torch.exp(-alpha_p_t * step_interval)  # [0-1]
            if not strict_traj:
                informative_mask[alpha.view(-1)>alpha_thres] = True
            else:
                uninformative_mask[alpha.view(-1)<alpha_thres] = True

            # [Optional] out-of-range particle
            # bbox_min = self.xyz_min.view(1, 3)*0.99
            # bbox_max = self.xyz_max.view(1, 3)*0.99
            # out_of_range_mask[(particle_pts_t<bbox_min).any(dim=1)] = True
            # out_of_range_mask[(particle_pts_t>bbox_max).any(dim=1)] = True

            # [Optional] check if p_t is within the occupied area
            # p_occ_t = self.grid_sampler(particle_pts_t, occ_grid)
            # p_alive_mark &= (p_occ_t.view(-1) > 0)

            # [optional] check if p_t is within the Background area
            if self.bg_mask_grid is not None:
                p_bg_t = self.grid_sampler(particle_pts_t, self.bg_mask_grid.float())
                in_bg_mask |=  (p_bg_t.view(-1) > 0.)

        # [1] remove `ever`/`always` uninformative particles
        if not strict_traj:  # [1-alternative] always uninformative
            uninformative_mask = ~informative_mask
        p_alive_mark[uninformative_mask] = False

        # [2] remove static particle
        k = int(0.9 * particle_traj_len.numel())
        percent_threshold = particle_traj_len.kthvalue(k).values.item()
        if percent_threshold < traj_thres:
            print(f"90% of particles have traj_len < {percent_threshold}, use this threshold.")
        
        threshold = min(traj_thres, percent_threshold)
        static_mask = particle_traj_len<threshold
        p_alive_mark[static_mask] = False

        # [3] remove invisible particle
        # p_occ_0 = self.grid_sampler(particle_pts_t_0, occ_grid)
        # p_alive_mark &= (p_occ_0.view(-1) > 0)

        # [4] remove out-of-range particle (clamped and lack some supervision)
        # p_alive_mark[out_of_range_mask] = False

        # [5] remove background particle
        if self.bg_mask_grid is not None:
            p_alive_mark[in_bg_mask] = False

        self.particle_pts_0 = torch.nn.Parameter(self.particle_pts_0[p_alive_mark])
        self.particle_feat = torch.nn.Parameter(self.particle_feat[p_alive_mark])
        self.particle_num = self.particle_pts_0.shape[0]

        p_str =  f'\033[96m Particle Removal: {old_particle_num}->{self.particle_num}. \033[0m'
        p_str += f'\033[96m uninformative({uninformative_mask.sum().item()})\033[0m'
        p_str += f'\033[96m static({static_mask.sum().item()})\033[0m'
        p_str += f'\033[96m out-of-range({out_of_range_mask.sum().item()})\033[0m'
        
        if self.bg_mask_grid is not None:
            p_str += f'\033[96m background({in_bg_mask.sum().item()})\033[0m'
        tqdm.write(p_str)
    

    @torch.no_grad()
    def reample_particles(self, resample_target):
        old_particle_num = self.particle_num
        extra_num = resample_target - old_particle_num
        if extra_num<=0:
            return
        # Q: resample at pts0 or pts_t_0??
        # A: use pts_t_0's sparsity, then sample pts0 (as particle signature) 
        # with very tiny shift (to enable randomness)
        particle_pts_t0 = self.particle_deform_t(torch.zeros(1, 1))

        # compute existing particle's sparsity
        sparsity_one = torch.ones(old_particle_num, 1)
        # TODO: can use larger voxel (relax resampling range)
        sparsity_grid = torch.zeros([1, 1, *(self.world_size//4)], dtype=torch.float32)
        sparsity_grid = self.particle_to_grid(particle_pts_t0, sparsity_one, sparsity_grid)
        sparsity_pts_t0 = 1. / (self.grid_sampler(particle_pts_t0, sparsity_grid) + 1e-3)
        sparsity_pts_t0 = sparsity_pts_t0/sparsity_pts_t0.sum()

        # compute particle(t=0)'s density
        alpha_p_t, _ = self.query_density(
            particle_pts_t0, torch.zeros(particle_pts_t0.shape[0], 1))
        density_prob = (alpha_p_t/alpha_p_t.sum()).view(-1)

        # The KEY to do resampling
        sample_prob = sparsity_pts_t0 * density_prob
        sample_prob = sample_prob/sample_prob.sum()

        # sampling around existing particle
        resample_idx = torch.multinomial(sample_prob, extra_num, replacement=True)
        resample_one = torch.ones(extra_num, dtype=torch.int)
        resample_num = scatter_add(resample_one, resample_idx, dim=0, dim_size=old_particle_num)
        # begin to sample
        resample_base = self.particle_pts_0
        resample_base_feat = self.particle_feat
        new_particle_base = torch.repeat_interleave(resample_base, resample_num, dim=0)
        new_particle_feat = torch.repeat_interleave(resample_base_feat, resample_num, dim=0)

        vox_sz = (self.xyz_max - self.xyz_min) / (self.world_size-1)
        resample_shift = (torch.rand_like(new_particle_base)*2-1) * vox_sz * 0.1
        new_particle_pts_0 = new_particle_base + resample_shift
        
        # replace
        particle_pts_0 = torch.cat([resample_base, new_particle_pts_0], dim=0).clone()
        particle_feat = torch.cat([resample_base_feat, new_particle_feat], dim=0).clone()
        self.particle_pts_0 = torch.nn.Parameter(particle_pts_0)
        self.particle_feat = torch.nn.Parameter(particle_feat)
        self.particle_num = self.particle_pts_0.shape[0]
        tqdm.write(f'\033[96m Particle Resample: {old_particle_num}->{self.particle_num}. \033[0m')


    def mult_dist_interp(self, ray_pts_delta, particle_pts_t=None):
        """
            INPUT: particle_pts_t[N_particle, 3]
                   ray_pts_delta [N_query_pts, 3]
        """
        grid = self.feature

        # particle to grid [added]
        if particle_pts_t is not None:
            grid = self.particle_to_grid(
                particle_pts_t, self.particle_feat, grid)
            self.fused_feature_grid = grid  # for latter regularisation
        
        # three scales
        vox_l = self.grid_sampler(ray_pts_delta, grid)
        # vox_m = self.grid_sampler(ray_pts_delta, grid[:,:,::2,::2,::2])
        # vox_s = self.grid_sampler(ray_pts_delta, grid[:,:,::4,::4,::4])
        # vox_feature = torch.cat((vox_l,vox_m,vox_s),-1)
        vox_feature = vox_l  # torch.cat((vox_l,vox_l,vox_l),-1)

        if len(vox_feature.shape)==1:
            vox_feature_flatten = vox_feature.unsqueeze(0)
        else:
            vox_feature_flatten = vox_feature
        
        return vox_feature_flatten

    def activate_density(self, density, interval=None): 
        interval = interval if interval is not None else self.voxel_size_ratio 
        return 1 - torch.exp(-F.softplus(density + self.act_shift) * interval) 

    def get_mask(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Check whether the rays hit the geometry or not'''
        shape = rays_o.shape[:-1]
        rays_o = rays_o.reshape(-1, 3).contiguous()
        rays_d = rays_d.reshape(-1, 3).contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id = render_utils_cuda.sample_pts_on_rays(
                rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)[:3]
        mask_inbbox = ~mask_outbbox
        hit = torch.zeros([len(rays_o)], dtype=torch.bool)
        hit[ray_id[mask_inbbox]] = 1
        return hit.reshape(shape)

    def particle_deform_t(self, times_sel):
        """
            times_sel[N_rays, 1], but synchronise time
        """
        times_emb = poc_fre(times_sel, self.time_poc)
        times_feature = self.timenet(times_emb)

        # particle deformation [added]
        particle_pts_emb = poc_fre(self.particle_pts_0, self.pos_poc)  # particle_pts_emb[:, :3] is particle_pts_0
        times_feature_particle = times_feature[0:1, :].expand(len(particle_pts_emb), -1)
        particle_pts_t = self.deformation_net(particle_pts_emb, times_feature_particle)

        return particle_pts_t

    def query_density(self, ray_pts, times_sel):
        # embedding: time, pos
        assert torch.all(times_sel.eq(times_sel[0])), "need to synchronise time"
        
        # # TODO: remove this coupling
        # experiments show that this only affect static part, so safe temporarily
        rays_pts_emb = poc_fre(ray_pts, self.pos_poc)

        # particle moving
        particle_pts_t = None
        if self.enabled_particle:
            particle_pts_t = self.particle_deform_t(times_sel)
            aabb_size = (self.xyz_max - self.xyz_min)
            particle_pts_t = torch.clamp(  # ensure within the bbox
                    particle_pts_t, min=self.xyz_max-(aabb_size*0.99), max=self.xyz_min+(aabb_size*0.99))
            self.particle_pts_t = particle_pts_t
        
        # voxel query interp
        vox_feature_flatten = self.mult_dist_interp(ray_pts, particle_pts_t)

        # out density
        vox_feature_flatten_emb = poc_fre(vox_feature_flatten, self.grid_poc)
        h_feature = self.featurenet(
            torch.cat(
                (vox_feature_flatten_emb, rays_pts_emb), -1  # times_feature
            )
        )
        density_result = self.densitynet(h_feature)

        # 1 - torch.exp(-F.softplus(density + self.act_shift) * interval) 
        alpha = nn.Softplus()(density_result + self.act_shift)

        # additional
        if self.dynamic_static_only is not None:
            particle_one = torch.ones(self.particle_num, 1)
            tmp_grid = torch.zeros([1, 1, *(self.world_size)], dtype=torch.float32)
            tmp_grid = self.particle_to_grid(particle_pts_t, particle_one, tmp_grid)
            dyn_mask = (self.grid_sampler(ray_pts, tmp_grid)>0.).view(-1)
            if self.dynamic_static_only == 'dynamic':
                alpha[~dyn_mask] = 0
            elif self.dynamic_static_only == 'static':
                alpha[dyn_mask] = 0
            else:
                raise NotImplementedError

        # if self.opaque_bkgd:
        #     alpha[..., -1, :] = torch.inf
        return alpha, h_feature

    def query_rgb_density(self, ray_pts, viewdirs, times_sel, cam_sel):
        # start from density-only function
        alpha, h_feature = self.query_density(ray_pts, times_sel)
        
        # further
        viewdirs_emb = poc_fre(viewdirs, self.view_poc)
        if self.add_cam == True:
            cam_emb = poc_fre(cam_sel, self.time_poc)
            cams_feature = self.camnet(cam_emb)
            viewdirs_emb = torch.cat((viewdirs_emb, cams_feature), -1)
        rgb_logit = self.rgbnet(h_feature, viewdirs_emb)
        rgb = torch.sigmoid(rgb_logit)

        return rgb, alpha

    def forward(
        self,
        rays_o,
        rays_d,
        viewdirs,
        times_sel,
        cam_sel=None,
        bg_points_sel=None,
        global_step=None,
        prop_requires_grad=False,
        **render_kwargs,
    ):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'

        # change render mode
        if 'dynamic_static_only' in render_kwargs:
            self.dynamic_static_only = render_kwargs['dynamic_static_only']

        ret_dict = {}
        N = len(rays_o)

        rays_d_normed = F.normalize(rays_d, dim=-1)

        def sigma_fn(t_starts, t_ends, ray_indices):
            if ray_indices.shape[0] == 0:
                return torch.zeros((0,), device=ray_indices.device)
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d_normed[ray_indices]
            t_times = times_sel[ray_indices]
            positions = (
                t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            )
            sigmas, _ = self.query_density(positions, t_times)
            return sigmas.squeeze(-1)

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            if ray_indices.shape[0] == 0:
                return torch.zeros(
                    (0, 3), device=ray_indices.device
                ), torch.zeros((0,), device=ray_indices.device)
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d_normed[ray_indices]
            t_views = viewdirs[ray_indices]
            t_times = times_sel[ray_indices]
            t_cams = None
            if self.add_cam:
                assert cam_sel is not None
                t_cams = cam_sel[ray_indices]
            positions = (
                t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            )
            rgbs, sigmas = self.query_rgb_density(
                positions, t_views, t_times, t_cams
            )
            return rgbs, sigmas.squeeze(-1)

        ray_indices, t_starts, t_ends = self.occ_grid.sampling(
            rays_o,
            rays_d_normed,
            sigma_fn=sigma_fn,
            near_plane=render_kwargs["near"],
            far_plane=render_kwargs["far"],
            render_step_size=render_kwargs["stepsize"] * self.voxel_size,
            stratified=self.training,
            alpha_thre=self.occ_alpha_thres,
        )
        rgb_marched, opacity, depth, extras = nerfacc.rendering(
            t_starts,
            t_ends,
            ray_indices=ray_indices,
            n_rays=N,
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_kwargs["bg"],
        )
        weights = extras["weights"]
        rgb = extras["rgbs"]
        alpha = extras["sigmas"]

        # filter nearly empty space's depth
        depth[opacity<0.1] = 0.

        ret_dict.update(
            {
                "alphainv_last": 1-opacity[:, 0],  # [N]
                "weights": weights,  # [S]
                "rgb_marched": rgb_marched,  # [N, 3]
                "raw_alpha": alpha,  # [S]
                "raw_rgb": rgb,  # [S, 3]
                "ray_id": ray_indices,  # [S]
                "depth": depth[:, 0],  # [N]
            }
        )

        # computer bg_points_delta
        if bg_points_sel is not None:
            pass
            # times_feature = self.timenet(
            #     poc_fre(
            #         torch.rand_like(bg_points_sel[:, :1]), self.time_poc
            #     )
            # )
            # bg_points_sel_emb = poc_fre(bg_points_sel, self.pos_poc)
            # bg_points_sel_delta = self.deformation_net(
            #     bg_points_sel_emb,
            #     times_feature,
            # )
            # ret_dict.update({"bg_points_delta": bg_points_sel_delta})
        
        return ret_dict


''' Ray and batch
'''
def get_rays(H, W, K, c2w, inverse_y=False, flip_x=False ,flip_y=False, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,3], np.shape(rays_d))
    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d

def get_rays_of_a_view(H, W, K, c2w, ndc, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs

@torch.no_grad()
def get_training_rays(rgb_tr, times,train_poses, HW, Ks, ndc):
    print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks),-1), axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(rgb_tr) == len(HW)
    H, W = HW[0]
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    times_tr = torch.ones([len(rgb_tr), H, W, 1], device=rgb_tr.device)

    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,)
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        times_tr[i] = times_tr[i]*times[i]
        del rays_o, rays_d, viewdirs
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, times_tr,rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_flatten(rgb_tr_ori, times,train_poses, HW, Ks, ndc):
    print('get_training_rays_flatten: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    times_tr=torch.ones([N,1], device=DEVICE)
    times=times.unsqueeze(-1)
    imsz = []
    top = 0
    for c2w, img, (H, W), K ,time_one in zip(train_poses, rgb_tr_ori, HW, Ks,times):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,)
        n = H * W
        times_tr[top:top+n]=times_tr[top:top+n]*time_one
        rgb_tr[top:top+n].copy_(img.flatten(0,1))
        rays_o_tr[top:top+n].copy_(rays_o.flatten(0,1).to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d.flatten(0,1).to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs.flatten(0,1).to(DEVICE))
        imsz.append(n)
        top += n
    assert top == N
    eps_time = time.time() - eps_time
    print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, times_tr,rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_in_maskcache_sampling(rgb_tr_ori, times,train_poses, HW, Ks, ndc, model, render_kwargs):
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    times_tr = torch.ones([N,1], device=DEVICE)
    times = times.unsqueeze(-1)
    imsz = []
    top = 0
    for c2w, img, (H, W), K ,time_one in zip(train_poses, rgb_tr_ori, HW, Ks,times):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,)
        mask = torch.empty(img.shape[:2], device=DEVICE, dtype=torch.bool)
        for i in range(0, img.shape[0], CHUNK):
            mask[i:i+CHUNK] = model.get_mask(
                    rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs).to(DEVICE)
        n = mask.sum()
        times_tr[top:top+n]=times_tr[top:top+n]*time_one
        rgb_tr[top:top+n].copy_(img[mask])
        rays_o_tr[top:top+n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs[mask].to(DEVICE))
        imsz.append(n)
        top += n

    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    eps_time = time.time() - eps_time
    print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, times_tr,rays_o_tr, rays_d_tr, viewdirs_tr, imsz


def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top:top+BS]
        top += BS

def batch_indices_generator_sameim(imsz, BS):
    while True:
        im_id = np.random.randint(0, len(imsz))
        idx_in_im = torch.LongTensor(np.random.randint(0, int(imsz[im_id]), BS))
        idx_global = idx_in_im + int(sum(imsz[:im_id]))
        yield idx_global

def poc_fre(input_data,poc_buf):

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb

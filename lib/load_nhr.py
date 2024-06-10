import warnings

warnings.filterwarnings("ignore")

import json
import os
import random
import numpy as np
import torch
import imageio, cv2


class Load_nhr_data():
    def __init__(self, datadir, ratio=0.5, white_bkgd=False, transform=None, palette_msk=False):
        self.datadir = datadir
        ann_file = os.path.join(datadir, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        cams = annots['cams']
        self.transform = None if (transform is None) else np.array(transform, dtype=np.float32)
        self.white_bkgd = white_bkgd
        self.palette_msk = palette_msk

        img_paths = np.array([
            np.array(ims_data['ims'])
            for ims_data in annots['ims']
        ])
        cam_inds = np.array([
            np.arange(len(ims_data['ims']))
            for ims_data in annots['ims']
        ])
        n_frame, n_view = img_paths.shape

        self.ratio = ratio
        self.img_paths = img_paths
        self.all_image = img_paths.ravel()
        self.all_image_camind = cam_inds.ravel()
        self.all_cams = cams
        im_time = np.zeros((n_frame, n_view))
        im_time[:, :] = np.linspace(0, 1, n_frame).reshape(-1, 1)
        self.all_time = im_time.ravel()
        
        # aabb
        pts = []
        for i in range(n_frame):
            vertices_path = os.path.join(datadir, 'vertices', '{}.npy'.format(i))
            if not os.path.exists(vertices_path):
                vertices_path = os.path.join(datadir, 'vertices', '{:06d}.npy'.format(i))
            wxyz = np.load(vertices_path).astype(np.float32)
            if self.transform is not None:
                wxyz = np.dot(wxyz, self.transform[:3, :3].T) + self.transform[:3, 3]
            pts.append(wxyz)
        pts = np.concatenate(pts, axis=0)
        self.aabb = np.concatenate([pts.min(axis=0), pts.max(axis=0)], axis=0)
        # TODO enlarge aabb by 0.1

        # near, far
        all_cam_o = np.stack(self.all_cams['T'], axis=0).squeeze() / 1000.
        dist = np.linalg.norm(all_cam_o[:, None] - all_cam_o, axis=-1)
        self.far = dist.max() * 0.8
        self.near = self.far * 0.2

        # following https://github.com/zju3dv/mlp_maps/blob/master/configs/nhr/sport1.py
        num_frame = 100
        test_interval = 10

        # train/test split
        # [0,9] removal is following mlp_maps[ref]
        total_id = np.arange(len(self.all_image)).reshape(n_frame, n_view)
        test_view = [0, 9, 18, 28, 37, 46]
        train_view = list(set(range(n_view)) - set(test_view))
        remove_view = [0, 9, 18] if self.palette_msk else [0, 9]  # special consideration of `basketball`
        test_view = [x for x in test_view if x not in remove_view]
        
        self.i_train = total_id[:num_frame, train_view].ravel()
        self.i_test = total_id[:num_frame:test_interval, test_view].ravel()

        # to update self.h, self.w
        self.h, self.w = None, None
        self.load_idx(0)

    
    def get_image(self, img_path, K, D):
        ori_img_path = img_path
        img_path = os.path.join(self.datadir, img_path)
        img = imageio.imread(img_path).astype(np.float32) / 255.

        def get_mask(data_root, img_path):
            msk_path = img_path.replace('images', 'mask')
            msk_path = os.path.join(data_root, msk_path)
            msk = imageio.imread(msk_path)[..., 0].astype(np.int32)
            msk = (msk > 100).astype(np.uint8)
            return msk
        
        img_path_to_schp_path = lambda x: 'schp' + x[6:-4] + '.png'
        def get_mask_basketball(data_root, img_path):
            msk_path = img_path_to_schp_path(img_path)
            msk_path = os.path.join(data_root, msk_path)
            msk = imageio.imread(msk_path)[..., :3].astype(np.int32)
            msk = (msk * [-1, 10, 100]).sum(axis=-1)

            palette = np.array([0, 128, 1280, 1408, 12800, 12928, 14080, 14208, 64,
                                192, 1344, 1472, 12864, 12992, 14144, 14272, 640, 768,
                                1920, 2048])
            leg_msk = (msk == palette[9]) | (msk == palette[16]) | (msk == palette[18])

            msk_path = img_path.replace('images', 'mask')
            msk_path = os.path.join(data_root, msk_path)
            msk = imageio.imread(msk_path)[..., 0].astype(np.int32)
            msk = (msk > 100).astype(np.uint8)
            msk[leg_msk] = 1
            return msk
        
        if self.palette_msk:
            msk = get_mask_basketball(self.datadir, ori_img_path)
        else:
            msk = get_mask(self.datadir, ori_img_path)
        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * self.ratio), int(img.shape[1] * self.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

        img[msk == 0] = 1 if self.white_bkgd else 0

        return img, msk

    def get_rays_torch(self, H, W, K, R, T):
        K = torch.from_numpy(K).float()
        R, T = torch.from_numpy(R).float(), torch.from_numpy(T).float()
        # calculate the camera origin
        rays_o = -torch.matmul(R.T, T).ravel()
        # calculate the world coodinates of pixels
        i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32),
                        torch.arange(H, dtype=torch.float32),
                        indexing='xy')
        # center_pixel:
        xy1 = torch.stack([i + 0.5, j + 0.5, torch.ones_like(i)], dim=2)
        xy1 = xy1.view(-1, 3).to(K)
        pixel_camera = torch.matmul(xy1, torch.linalg.inv(K).T)
        pixel_world = torch.matmul(pixel_camera - T.ravel(), R)
        # calculate the ray direction
        rays_d = pixel_world - rays_o[None, None]
        rays_d = rays_d / torch.linalg.norm(rays_d, dim=2, keepdims=True)
        rays_o = torch.broadcast_to(rays_o, rays_d.shape)
        return rays_o, rays_d


    def load_idx(self, idx):
        img_path = self.all_image[idx]
        cam_ind = self.all_image_camind[idx]
        K = np.array(self.all_cams['K'][cam_ind])
        D = np.array(self.all_cams['D'][cam_ind])
        R = np.array(self.all_cams['R'][cam_ind])
        T = np.array(self.all_cams['T'][cam_ind]) / 1000.
        RT = np.concatenate([R, T], axis=1)
        RT = np.concatenate([RT, [[0, 0, 0, 1]]], axis=0)
        if self.transform is not None:
            # transform = np.eye(4)   # NOTE: if do transform, revise self.aabb computation
            RT = np.dot(RT, np.linalg.inv(self.transform))
            R, T = RT[:3, :3], RT[:3, 3:]
        
        img, msk = self.get_image(img_path, K, D)
        H, W = img.shape[:2]
        K[:2] = K[:2] * self.ratio

        rays_o, rays_d = self.get_rays_torch(H, W, K, R, T)
        rays_color = torch.from_numpy(img).float().view(-1,3)

        # there are two types of sizes, so dynamically update.
        self.h, self.w = H, W

        return {'rays_ori': rays_o.squeeze(), 
                'rays_dir': rays_d.squeeze(), 
                'viewdirs':rays_d.squeeze(),
                'rays_color': rays_color, 
                'near': torch.tensor(self.near).float().view(-1), 
                'far': torch.tensor(self.far).float().view(-1),
                'H': H, 'W': W, 'K': K, 'R': R, 'T': T}

_base_ = './default.py'

expname = 'nerf-p/dnerf_lego-400'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/data/datasets/nerf/d_nerf/lego',
    dataset_type='dnerf',
    white_bkgd=True,
    aabb=(-1.3, -1.3, -1.3, 1.3, 1.3, 1.3),
)

train_config = dict(
    N_iters=60000,
    pg_scale=[2000, 4000, 6000],

    # particle
    init_particle_after=2000,
    lrate_deformation_net=6e-4,
    lrate_particle_pts_0=6e-4,
    lrate_particle_feat=1e-2,

    # removal
    remove_particle_steps=list(range(4000, 20000, 2000)),
    remove_particle_traj_thres=0.01,
    remove_particle_alpha_thres=[0.1, 0.1],  # [before_the_last, the_last],
    # resample
    resample_target=200000
)

model_and_render = dict(
    # occupancy mask
    occ_grid_reso=128,
    occ_ema_decay=0.99,

    # particle
    particle_num=200000,
    voxel_dim=12,
)
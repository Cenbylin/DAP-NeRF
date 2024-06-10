_base_ = "./hyper_default.py"

expname = "vrig-p/peel-banana"
basedir = "./logs/vrig_data"

data = dict(
    datadir="./data/vrig/vrig-peel-banana",
    dataset_type="hyper_dataset",
    white_bkgd=False,
    use_bg_points=False,  # our advantage
)

train_config = dict(
    N_iters=60000,
    pg_scale=[2000, 6000, 8000, 12000],

    # particle
    init_particle_after=3000,
    lrate_deformation_net=1e-3,
    lrate_particle_pts_0=1e-3,
    lrate_particle_feat=1e-2,
    lrate_decay=20,

    # removal
    remove_particle_steps=list(range(4000, 30000, 2000)),
    remove_particle_traj_thres=0.1,
    remove_particle_alpha_thres=[0.0001, 0.001],  # [before_the_last, the_last],
    # resample
    resample_target=500000,
    
    tv_every=5,
    tv_after=3000,
    tv_before=1e9,
    weight_tv_motion=0.1,
    weight_tv_feature=0.05,
    
    # added
    strict_traj=True,
)

model_and_render = dict(
    occ_grid_reso=128,
    occ_ema_decay=0.99,
    occ_thre=0.05,

    # particle
    particle_num=500000,
    voxel_dim=16,
)

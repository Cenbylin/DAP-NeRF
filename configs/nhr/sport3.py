c_base_ = './default.py'

expname = 'nhr-p/sport3'
basedir = './logs/nhr'

data = dict(
    datadir='./data/NHR/sport_3_easymocap',
    dataset_type='nhr',
    white_bkgd=True,
    palette_msk=False,
    transform=[[ 0.98480775, -0.04494346,  0.16773126, -0.75454014],
               [-0.17364818, -0.254887  ,  0.95125124, -4.9550868 ],
               [ 0.        , -0.96592583, -0.25881905,  1.9544816 ],
               [ 0.        ,  0.        ,  0.        ,  1.        ]],
    # aabb=(-1.3, -1.3, -1.3, 1.3, 1.3, 1.3),
)

train_config = dict(
    N_iters=60000,
    pg_scale=[2000, 6000, 10000],

    # particle
    init_particle_after=3000,
    lrate_deformation_net=3e-4,
    lrate_particle_pts_0=3e-4,
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
    weight_tv_motion=0.01,
    weight_tv_feature=0.01,
)

model_and_render = dict(
    # occupancy mask
    occ_grid_reso=128,
    occ_ema_decay=0.99,

    # particle
    particle_num=500000,
    voxel_dim=16,
)
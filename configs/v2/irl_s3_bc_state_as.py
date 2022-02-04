_base_ = ['../_base_/v2/irl_s3_state.py']


env_cfg = dict(
    type='gym',
    env_name='OpenCabinetDrawer_1045_link_0-v0',
    extra_wrappers=dict(type='IRLASWrapper',env=None)
)

agent = dict(
    type='BCStateSB',
    batch_size=32,
    policy_cfg=dict(
        gail_cfg = dict(
            total_timesteps = 2000*1024,
            irl_timesteps = 2000*1024//3,
            resume = False,
            save = True,
            policy_model = "policy",
            reward_model = "reward",
            algo = "bc",
            gen_algo = 'mlp',
            true_reward = False,
            pretrain = False,
            lr_update=False,
            sac_algo = dict(
                buffer_size = 2000000,
                learning_rate = 0.0003,
                learning_starts = 5*1024,
                batch_size = 1024,
                gamma = 0.95,
                verbose = 1,
                seed = 10,
                policy = 'MlpPolicy'
            ),
            irl_algo = dict(
                demo_batch_size = 25*1000,#15*num_of_traj
                n_disc_updates_per_round=5,
                init_tensorboard = True,
                gen_train_timesteps = 50*1024,
                init_tensorboard_graph = True,
                allow_variable_horizon = True,
                normalize_obs=True,
                normalize_reward=True,
                disc_opt_kwargs=dict(
                    lr= 1e-3,
                    # betas=(0.9, 0.999)
                    ),
            ),
            bc_config = dict(
                init = dict(
                    batch_size = 1024,
                ),
                train = dict(
                    n_epochs = 50*2*2 , 
                    # n_batches = 2*1024
                )
            )
        ),
    ),
)

# For single horizon state
# replay_cfg = dict(
#     type='TrajReplayStateABS',
#     capacity=2000000,
#     horizon=1
# )

# For finite horizon state
replay_cfg = dict(
    type='TrajReplayStateABS',
    capacity=2000000,
    horizon=1
)

finiteH_as = False

if finiteH_as:
    agent = dict(
    policy_cfg=dict(
        gail_cfg = dict(
            finiteH_as=True,
        ),
        )
    )
    replay_cfg = dict(
        type='TrajReplayStateABS',
        capacity=2000000,
        horizon=200
    )

train_mfrl_cfg = dict(
    total_steps=1,
    warm_steps=0,
    n_steps=0,
    n_updates=1,
    n_eval=1,
    n_checkpoint=1,
    init_replay_buffers='./example_mani_skill_data/OpenCabinetDrawer_1045_link_0-v0_pcd.h5',
)

eval_cfg = dict(
    type='Evaluation',
    num=10,
    num_procs=1,
    use_hidden_state=False,
    start_state=None,
    save_traj=True,
    save_video=True,
    use_log=False,
)

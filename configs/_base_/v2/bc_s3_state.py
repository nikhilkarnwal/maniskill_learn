log_level = 'INFO'
stack_frame = 1
num_heads = 4

agent = dict(
    type='IRLStateSB',
    batch_size=32,
    policy_cfg=dict(
        gail_cfg = dict(
            total_timesteps = 3000*1024*2,
            irl_timesteps = 3000*1024*2//1,
            resume = False,
            save = True,
            policy_model = "policy",
            reward_model = "reward",
            algo = "airl",
            gen_algo = 'ppo',
            true_reward = False,
            pretrain = False,
            pretrain_ts = 100000,
            lr_update=True,
            save_freq=100000,
            eval_freq=20000,
            explore=True,
            disc_lr_exp=0.95,
            sac_algo = dict(
                buffer_size = 600000,
                learning_rate = 0.0003,
                learning_starts = 4*1024,
                batch_size = 1024,
                gamma = 0.95,
                verbose = 1,
                policy = 'MlpPolicy',
                tau = 0.01
            ),
            ppo_algo = dict(
                learning_rate = 0.0003,
                batch_size = 1024,
                n_steps = 1024*8,
                verbose = 1,
                policy = 'MlpPolicy',
            ),
            bc_config = dict(
                init=dict(
                    batch_size = 32,
                ),
                train = dict(
                    n_epochs = 100,
                )
            )
            ,
            irl_algo = dict(
                demo_batch_size = 1024,#15*num_of_traj
                n_disc_updates_per_round=1,
                init_tensorboard = True,
                gen_train_timesteps = 1024*4*8*2,
                init_tensorboard_graph = True,
                allow_variable_horizon = True,
                # normalize_obs=False,
                # normalize_reward=False,
                # disc_opt_kwargs=dict(
                #     lr= 1e-3,
                #     # betas=(0.95, 0.999),
                #     # weight_decay=1e-3,
                #     ),
            )
        ),
    ),
)

# eval_cfg = dict(
#     type='Evaluation',
#     num=10,
#     num_procs=1,
#     use_hidden_state=False,
#     start_state=None,
#     save_traj=True,
#     save_video=True,
#     use_log=False,
# )

train_mfrl_cfg = dict(
    on_policy=False,
)

env_cfg = dict(
    type='gym',
    unwrapped=False,
    stack_frame=stack_frame,
    obs_mode='state',
    reward_type='dense'
)

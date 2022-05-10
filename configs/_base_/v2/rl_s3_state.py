log_level = 'INFO'
stack_frame = 1
num_heads = 4

agent = dict(
    type='RLSB',
    batch_size=32,
    policy_cfg=dict(
        type='ContinuousPolicy',
        policy_head_cfg=dict(
            type='DeterministicHead',
            noise_std=1e-5,
        ),
        gail_cfg = dict(
            demo_batch_size = 64,
            timesteps = 2048,
            total_timesteps = 2*1000000,
            resume = False,
            save = True,
            policy_model = "policy",
            gen_algo = 'sac',
            sac_algo = dict(
                buffer_size = 1000000,
                learning_rate = 0.0003,
                learning_starts = 4000,
                batch_size = 1024,
                gamma = 0.95,
                verbose = 1,
                policy = 'MlpPolicy'
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

_base_ = ['../_base_/v2/irl_s3_state.py']


env_cfg = dict(
    type='gym',
    env_name='OpenCabinetDrawer_1045_link_0-v0',
    extra_wrappers=dict(type='IRLASWrapper',env=None)
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

finiteH_as = 2

if finiteH_as > 0:
    agent = dict(
    policy_cfg=dict(
        gail_cfg = dict(
            finiteH_as=2,
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

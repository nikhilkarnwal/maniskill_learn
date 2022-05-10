_base_ = ['../_base_/v2/irl_s3_state.py']


env_cfg = dict(
    type='gym',
    env_name='OpenCabinetDrawer_1045_link_0-v0',
    # extra_wrappers=dict(type='IRLFHWrapper',env=None)
)

replay_cfg = dict(
    type='TrajReplayState',
    capacity=2000000,
)


agent = dict(
policy_cfg=dict(
    gail_cfg = dict(
        finiteH_as=0,
    ),
    )
)

train_mfrl_cfg = dict(
    total_steps=1,
    warm_steps=0,
    n_steps=0,
    n_updates=1,
    n_eval=1,
    n_checkpoint=1,
    init_replay_buffers=["/media/biswas/D/maniskill/rl/v2/sac/env1000_link_0/rl_reach_door/RLSB/test/trajectory.h5"],
)

eval_cfg = dict(
    type='Evaluation',
    num=100,
    num_procs=1,
    use_hidden_state=False,
    start_state=None,
    save_traj=True,
    save_video=True,
    use_log=False,
)

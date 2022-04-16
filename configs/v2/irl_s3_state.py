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
    init_replay_buffers=['/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1000_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1026_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1068_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1027_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1054_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1073_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1001_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1028_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1044_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1057_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1075_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1002_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1030_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1006_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1031_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1077_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1007_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1034_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1061_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1078_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1045_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1062_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1014_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1036_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1046_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1063_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1081_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1017_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1047_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1018_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1038_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1049_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1039_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1051_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1065_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1041_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1052_link_0-v0.h5',
                        '/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1067_link_0-v0.h5'],
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

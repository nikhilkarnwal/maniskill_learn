#!/bin/bash
# change the environment name to OpenCabinetDoor, OpenCabinetDrawer, PushChair, or MoveBucket
# change the network config
# increase eval_cfg.num_procs for parallel evaluation

model_list=$(python -c "import mani_skill, os, os.path as osp; print(osp.abspath(osp.join(osp.dirname(mani_skill.__file__), 'assets', 'config_files', 'cabinet_models_drawer.yml')))")


python -m tools.run_rl configs/bc/mani_skill_gail_point_cloud_transformer.py --gpu-ids=0 \
	--work-dir=./work_dirs/gail_drawer/ \
	--cfg-options "train_mfrl_cfg.total_steps=1" "train_mfrl_cfg.init_replay_buffers=" \
	"train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_data/OpenCabinetDrawer/\",\"$model_list\"]" \
	"env_cfg.env_name=OpenCabinetDrawer-v0"  "train_mfrl_cfg.n_eval=1" \
	--resume-from=./full_mani_skill_data/models/OpenCabinetDrawer-v0_PN_Transformer.ckpt

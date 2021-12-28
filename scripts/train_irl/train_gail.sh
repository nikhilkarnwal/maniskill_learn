#!/bin/bash
# change the environment name to OpenCabinetDoor, OpenCabinetDrawer, PushChair, or MoveBucket
# change the network config
# increase eval_cfg.num_procs for parallel evaluation

model_list=$(python -c "import mani_skill, os, os.path as osp; print(osp.abspath(osp.join(osp.dirname(mani_skill.__file__), 'assets', 'config_files', 'cabinet_models_drawer.yml')))")


# python -m tools.run_rl configs/bc/mani_skill_gail_point_cloud_transformer.py --gpu-ids=0 \
# 	--work-dir=./work_dirs/sac_drawer_single_obj/ \
# 	--cfg-options "train_mfrl_cfg.total_steps=1" "train_mfrl_cfg.init_replay_buffers=" \
# 	"train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_data/OpenCabinetDrawer/\",\"$model_list\"]" \
# 	"env_cfg.env_name=OpenCabinetDrawer_1000-v0"  "train_mfrl_cfg.n_eval=1" \
# 	--resume-from=./full_mani_skill_data/models/OpenCabinetDrawer-v0_PN_Transformer.ckpt



# # Train gail using states only
# python -m tools.run_rl configs/v2/irl_s3_state.py --gpu-ids=0 \
# 	--work-dir=./work_dirs/irl_drawer/irl/gail_sac_env1000/ \
# 	--cfg-options "train_mfrl_cfg.total_steps=1" \
# 	"env_cfg.env_name=OpenCabinetDrawer_1000-v0" "train_mfrl_cfg.init_replay_buffers=" \
# 	"train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_state_data/OpenCabinetDrawer_state/\",\"$model_list\"]" \
# 	"agent.policy_cfg.gail_cfg.algo=gail"


# train using states only

replay_buff="./full_mani_skill_state_data/OpenCabinetDrawer_state/custom_1000_link0_v0.h5"
# , \"./full_mani_skill_state_data/OpenCabinetDrawer_state/OpenCabinetDrawer_1000_link_1-v0.h5"
c_t=$(date "+%d_%m_%Y_%H_%M_%S")
mkdir ./work_dirs/irl_drawer/irl/v0/gail_sac_env1000/$c_t/
echo $1 > ./work_dirs/irl_drawer/irl/v0/gail_sac_env1000/$c_t/desc.txt
# #train airl
# python -m tools.run_rl configs/v2/irl_s3_state.py --gpu-ids=0 \
# 	--work-dir=./work_dirs/irl_drawer/irl/v0/airl_sac_env1000/ \
# 	--cfg-options "train_mfrl_cfg.total_steps=1" \
# 	"env_cfg.env_name=OpenCabinetDrawer_1000-v0" "train_mfrl_cfg.init_replay_buffers=$replay_buff" \
# 	"agent.policy_cfg.gail_cfg.algo=airl"
# 	# "train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_state_data/OpenCabinetDrawer_state/\",\"$model_list\"]" \

# # train gail
# python -m tools.run_rl configs/v2/irl_s3_state.py --gpu-ids=0 \
# 	--work-dir=./work_dirs/irl_drawer/irl/v0/gail_sac_env1000/$c_t/ \
# 	--cfg-options "train_mfrl_cfg.total_steps=1" \
# 	"env_cfg.env_name=OpenCabinetDrawer_1000_link_0-v0" "train_mfrl_cfg.init_replay_buffers=[$replay_buff]" \
# 	"agent.policy_cfg.gail_cfg.algo=gail"
# 	# "train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_state_data/OpenCabinetDrawer_state/\",\"$model_list\"]" \


# train gail with as
python -m tools.run_rl configs/v2/irl_s3_state_as.py --gpu-ids=0 \
	--work-dir=./work_dirs/irl_drawer/irl/v0/gail_sac_env1000/$c_t/ \
	--cfg-options "train_mfrl_cfg.total_steps=1" \
	"env_cfg.env_name=OpenCabinetDrawer_1000_link_0-v0" "train_mfrl_cfg.init_replay_buffers=[$replay_buff]" \
	"agent.policy_cfg.gail_cfg.algo=gail"
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

#====================================Door======================================#

# replay_buff = "['/media/biswas/D/maniskill/data/door/state/OpenCabinetDoor_state/OpenCabinetDoor_1000_link_0-v0.h5',
for itr in 1 2 3
do
    echo "Running $itr Iteration"

	c_t=$(date "+%d_%m_%Y_%H_%M_%S")
	model_dir="/media/biswas/D/maniskill/IRLStateSB/gail/sac/"
	base_dir="/media/biswas/D/maniskill"
	work_dir="work_dirs/irl_door/irl/v0/gail_sac_env1000"

	# replay_buff="$base_dir/rl/v2/sac/env1000_link_0/finite_horizon_data/trajectory.h5"
	mkdir -p $base_dir/$work_dir/$c_t/
	echo $1 > $base_dir/$work_dir/$c_t/desc.txt
	# #train airl
	# python -m tools.run_rl configs/v2/irl_s3_state.py --gpu-ids=0 \
	# 	--work-dir=./work_dirs/irl_drawer/irl/v0/airl_sac_env1000/ \
	# 	--cfg-options "train_mfrl_cfg.total_steps=1" \
	# 	"env_cfg.env_name=OpenCabinetDrawer_1000-v0" "train_mfrl_cfg.init_replay_buffers=$replay_buff" \
	# 	"agent.policy_cfg.gail_cfg.algo=airl"
	# 	# "train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_state_data/OpenCabinetDrawer_state/\",\"$model_list\"]" \

	# # train gail
	# python -m tools.run_rl configs/v2/irl_s3_state.py --gpu-ids=0 \
	# 	--work-dir=$base_dir/$work_dir/$c_t/ \
	# 	--cfg-options "train_mfrl_cfg.total_steps=1" \
	# 	"env_cfg.env_name=OpenCabinetDoor_1000_link_0-v0" \
	# 	"agent.policy_cfg.gail_cfg.algo=gail" "replay_cfg.capacity=3000000"  >  $base_dir/$work_dir/$c_t/temp_log.txt 
	# 	# "agent.policy_cfg.gail_cfg.resume_model=$model_dir/24_12_2021_17_25_56/sac_policy.zip" \
	# 	# "agent.policy_cfg.gail_cfg.reward_model=$model_dir/24_12_2021_17_25_56/reward" \
	# 	# "agent.policy_cfg.gail_cfg.resume=1" 

	#=====================================================Reach======================================================#
	# train gail
	python -m tools.run_rl configs/v2/irl_s3_state_reach.py --gpu-ids=0 \
		--work-dir=$base_dir/$work_dir/$c_t/ \
		--cfg-options "train_mfrl_cfg.total_steps=1" \
		"env_cfg.env_name=OpenCabinetDoorReach-v0" \
		"agent.policy_cfg.gail_cfg.algo=gail" "replay_cfg.capacity=3000000"  >  $base_dir/$work_dir/$c_t/temp_log.txt 
		# "agent.policy_cfg.gail_cfg.resume_model=$model_dir/24_12_2021_17_25_56/sac_policy.zip" \
		# "agent.policy_cfg.gail_cfg.reward_model=$model_dir/24_12_2021_17_25_56/reward" \
		# "agent.policy_cfg.gail_cfg.resume=1" 

	# # train airl
	# python -m tools.run_rl configs/v2/irl_s3_state_reach.py --gpu-ids=0 \
	# 	--work-dir=$base_dir/$work_dir/$c_t/ \
	# 	--cfg-options "train_mfrl_cfg.total_steps=1" \
	# 	"env_cfg.env_name=OpenCabinetDoorReach-v0" \
	# 	"agent.policy_cfg.gail_cfg.algo=airl" "agent.policy_cfg.gail_cfg.gen_algo=ppo" "replay_cfg.capacity=3000000"  >  $base_dir/$work_dir/$c_t/temp_log.txt 
	# 	# "agent.policy_cfg.gail_cfg.resume_model=$model_dir/24_12_2021_17_25_56/sac_policy.zip" \
	# 	# "agent.policy_cfg.gail_cfg.reward_model=$model_dir/24_12_2021_17_25_56/reward" \
	# 	# "agent.policy_cfg.gail_cfg.resume=1" 
	#====================================================================Drawer======================================#

	# replay_buff="./full_mani_skill_state_data/OpenCabinetDrawer_state/custom_1000_link0_v0_v1.h5"
	# replay_buff="./full_mani_skill_state_data/OpenCabinetDrawer_state/OpenCabinetDrawer_1000_link_0-v0.h5"
	# c_t=$(date "+%d_%m_%Y_%H_%M_%S")
	# model_dir="/media/biswas/D/maniskill/IRLStateSB/gail/sac/"
	# base_dir="/media/biswas/D/maniskill"
	# work_dir="work_dirs/irl_drawer/irl/v0/gail_sac_env1000"

	# # replay_buff="$base_dir/rl/v2/sac/env1000_link_0/finite_horizon_data/trajectory.h5"
	# mkdir -p $base_dir/$work_dir/$c_t/
	# echo $1 > $base_dir/$work_dir/$c_t/desc.txt
	# # #train airl
	# # python -m tools.run_rl configs/v2/irl_s3_state.py --gpu-ids=0 \
	# # 	--work-dir=./work_dirs/irl_drawer/irl/v0/airl_sac_env1000/ \
	# # 	--cfg-options "train_mfrl_cfg.total_steps=1" \
	# # 	"env_cfg.env_name=OpenCabinetDrawer_1000-v0" "train_mfrl_cfg.init_replay_buffers=$replay_buff" \
	# # 	"agent.policy_cfg.gail_cfg.algo=airl"
	# # 	# "train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_state_data/OpenCabinetDrawer_state/\",\"$model_list\"]" \

	# # train gail
	# python -m tools.run_rl configs/v2/irl_s3_state.py --gpu-ids=0 \
	# 	--work-dir=$base_dir/$work_dir/$c_t/ \
	# 	--cfg-options "train_mfrl_cfg.total_steps=1" \
	# 	"env_cfg.env_name=OpenCabinetDrawer_1000_link_0-v0" "train_mfrl_cfg.init_replay_buffers=[$replay_buff]" \
	# 	"agent.policy_cfg.gail_cfg.algo=gail" "replay_cfg.capacity=3000000"  >  $base_dir/$work_dir/$c_t/temp_log.txt 
	# 	# "agent.policy_cfg.gail_cfg.resume_model=$model_dir/24_12_2021_17_25_56/sac_policy.zip" \
	# 	# "agent.policy_cfg.gail_cfg.reward_model=$model_dir/24_12_2021_17_25_56/reward" \
	# 	# "agent.policy_cfg.gail_cfg.resume=1" 

	# # Eval gail
	# python -m tools.run_rl configs/v2/irl_s3_state.py --gpu-ids=0 --evaluation \
	# 	--work-dir=$base_dir/$work_dir/$c_t/ \
	# 	--cfg-options "train_mfrl_cfg.total_steps=1" \
	# 	"env_cfg.env_name=OpenCabinetDrawer_1000_link_0-v0" "eval_cfg.num=1000" \
	# 	"eval_cfg.num_procs=1" "eval_cfg.use_log=True" "eval_cfg.save_traj=True" \
	# 	"agent.policy_cfg.gail_cfg.algo=gail" "replay_cfg.capacity=3000000"  \
	# 	"agent.policy_cfg.gail_cfg.resume_model=$model_dir/22_02_2022_16_56_32/rl_model_300000_steps.zip" \
	# 	"agent.policy_cfg.gail_cfg.resume=1" \
	# 	--resume-from ./full_mani_skill_data/models/OpenCabinetDrawer-v0_PN_Transformer.ckpt 


	# # train gail with as
	# python -m tools.run_rl configs/v2/irl_s3_state_as.py --gpu-ids=0 \
	# 	--work-dir=$base_dir/$work_dir/$c_t/ \
	# 	--cfg-options "train_mfrl_cfg.total_steps=1" \
	# 	"env_cfg.env_name=OpenCabinetDrawer_1000_link_0-v0" "train_mfrl_cfg.init_replay_buffers=[$replay_buff]" \
	# 	"agent.policy_cfg.gail_cfg.algo=gail" \
	# 	"agent.policy_cfg.gail_cfg.pretrain=0" "agent.policy_cfg.gail_cfg.pretrain_ts=60000" \
	# 	# "agent.policy_cfg.gail_cfg.resume_model=$model_dir/03_01_2022_16_28_16/best_model.zip" \
	# 	# "agent.policy_cfg.gail_cfg.reward_model=None" \
	# 	# "agent.policy_cfg.gail_cfg.resume=1" 


	# sh scripts/train_irl/train_gail.sh "lr-disc=1e-3,lrupdate, custom traj=1000, steps=4e6, gen_step=25k, batch=25k,disc_round-2,fhabs=2,pretrain=0"
done
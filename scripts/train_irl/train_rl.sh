c_t=$(date "+%d_%m_%Y_%H_%M_%S")
base_dir="/media/biswas/D/maniskill"
work_dir="rl/v2/sac/env1000_link_0"
mkdir -p $base_dir/$work_dir/$c_t/
echo $1 > $base_dir/$work_dir/$c_t/desc.txt

# python -m tools.run_rl configs/v2/rl_s3_state.py --gpu-ids=0 \
# 	--work-dir=$base_dir/$work_dir/$c_t/ \
# 	--cfg-options "train_mfrl_cfg.total_steps=1" \
# 	"env_cfg.env_name=OpenCabinetDoorReach-v0"

python -m tools.run_rl configs/v2/rl_s3_state.py --evaluation --gpu-ids=0 \
--work-dir=$base_dir/$work_dir/$c_t/ \
--cfg-options "env_cfg.env_name=OpenCabinetDoorReach-v0" "eval_cfg.num=1000" "eval_cfg.num_procs=1" "eval_cfg.use_log=True" "eval_cfg.save_traj=True" \
"agent.policy_cfg.gail_cfg.resume_model=26_04_2022-09_10_12_sac_policy" \
"agent.policy_cfg.gail_cfg.reward_model=None" \
"agent.policy_cfg.gail_cfg.resume=1" \
--resume-from="./full_mani_skill_data/models/OpenCabinetDrawer-v0_PN_Transformer.ckpt"

    # python -m tools.run_rl configs/bc/mani_skill_gail_point_cloud_transformer.py --gpu-ids=0 \
	# --work-dir=./work_dirs/sac_drawer_single_obj/ \
	# --cfg-options "train_mfrl_cfg.total_steps=1" "train_mfrl_cfg.init_replay_buffers=" \
	# "train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_data/OpenCabinetDrawer/\",\"$model_list\"]" \
	# "env_cfg.env_name=OpenCabinetDrawer_1000-v0"  "train_mfrl_cfg.n_eval=1" \
	# --resume-from=./full_mani_skill_data/models/OpenCabinetDrawer-v0_PN_Transformer.ckpt

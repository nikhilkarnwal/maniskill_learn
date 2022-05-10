# train using states only

# replay_buff="./full_mani_skill_state_data/OpenCabinetDrawer_state/custom_1000_link0_v0.h5"
replay_buff="/media/biswas/D/maniskill/rl/v2/sac/env1000_link_0/rl_reach_door/RLSB/test/trajectory.h5"
# , \"./full_mani_skill_state_data/OpenCabinetDrawer_state/OpenCabinetDrawer_1000_link_1-v0.h5"
c_t=$(date "+%d_%m_%Y_%H_%M_%S")
base_dir="/media/biswas/D/maniskill"
work_dir="work_dirs/irl_drawer/irl/v0/bc_env1000"
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
# 	--work-dir=./work_dirs/irl_drawer/irl/v0/gail_sac_env1000/$c_t/ \
# 	--cfg-options "train_mfrl_cfg.total_steps=1" \
# 	"env_cfg.env_name=OpenCabinetDrawer_1000_link_0-v0" "train_mfrl_cfg.init_replay_buffers=[$replay_buff]" \
# 	"agent.policy_cfg.gail_cfg.algo=gail"
# 	# "train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_state_data/OpenCabinetDrawer_state/\",\"$model_list\"]" \


# # train gail with as
# python -m tools.run_rl configs/v2/irl_s3_bc_state_as.py --gpu-ids=0 \
# 	--work-dir=$base_dir/$work_dir/$c_t/ \
# 	--cfg-options "train_mfrl_cfg.total_steps=1" \
# 	"env_cfg.env_name=OpenCabinetDrawer_1000_link_0-v0" "train_mfrl_cfg.init_replay_buffers=[$replay_buff]" \
# 	"agent.policy_cfg.gail_cfg.algo=gail"

# train gail with as
python -m tools.run_rl configs/v2/irl_s3_bc_state_as.py --gpu-ids=0 \
	--work-dir=$base_dir/$work_dir/$c_t/ \
	--cfg-options "train_mfrl_cfg.total_steps=1" \
	"env_cfg.env_name=OpenCabinetDoorReach-v0" "train_mfrl_cfg.init_replay_buffers=[$replay_buff]" \
	"agent.policy_cfg.gail_cfg.algo=gail" "eval_cfg.num=100"
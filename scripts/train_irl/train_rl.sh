c_t=$(date "+%d_%m_%Y_%H_%M_%S")
mkdir ./work_dirs/irl_drawer/rl/v2/sac/$c_t
echo $1 > ./work_dirs/irl_drawer/rl/v2/sac/$c_t/desc.txt

python -m tools.run_rl configs/v2/rl_s3_state.py --gpu-ids=0 \
	--work-dir=./work_dirs/irl_drawer/rl/v2/sac/$c_t \
	--cfg-options "train_mfrl_cfg.total_steps=1" \
	"env_cfg.env_name=OpenCabinetDrawer_1000_link_0-v0"


    # python -m tools.run_rl configs/bc/mani_skill_gail_point_cloud_transformer.py --gpu-ids=0 \
	# --work-dir=./work_dirs/sac_drawer_single_obj/ \
	# --cfg-options "train_mfrl_cfg.total_steps=1" "train_mfrl_cfg.init_replay_buffers=" \
	# "train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_data/OpenCabinetDrawer/\",\"$model_list\"]" \
	# "env_cfg.env_name=OpenCabinetDrawer_1000-v0"  "train_mfrl_cfg.n_eval=1" \
	# --resume-from=./full_mani_skill_data/models/OpenCabinetDrawer-v0_PN_Transformer.ckpt
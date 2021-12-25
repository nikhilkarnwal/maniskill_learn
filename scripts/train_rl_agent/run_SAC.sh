# python -m tools.run_rl configs/sac/sac_mani_skill_state_1M_train.py --seed=0 --cfg-options "env_cfg.env_name=OpenCabinetDrawer_1000_link_0-v0" --gpu-ids=0 --clean-up
# model_list=$(python -c "import mani_skill, os, os.path as osp; print(osp.abspath(osp.join(osp.dirname(mani_skill.__file__), 'assets', 'config_files', 'cabinet_models_drawer.yml')))")


# python -m tools.run_rl configs/sac/sac_mani_skill_state_1M_train.py --gpu-ids=0 \
# 	--work-dir=./work_dirs/sac_transformer_drawer/ \
# 	--cfg-options "env_cfg.env_name=OpenCabinetDrawer-v0" "eval_cfg.save_video=True" 
	#--resume-from ./work_dirs/sac_transformer_drawer/SAC1/models/model_1000000.ckpt
	# "train_mfrl_cfg.total_steps=5000" "train_mfrl_cfg.n_steps=8" "train_mfrl_cfg.init_replay_buffers=" \
	# "train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_data/OpenCabinetDrawer/\",\"$model_list\"]" \
	# "env_cfg.env_name=OpenCabinetDrawer-v0" "eval_cfg.num=10" "eval_cfg.num_procs=1" "train_mfrl_cfg.n_eval=10" "eval_cfg.save_video=True"



python -m tools.run_rl configs/sac/sac_mani_skill_state_1M_train.py --evaluation --gpu-ids=0 \
--work-dir=./test/sac_transformer_drawer/v0/env1000_link0/ \
--resume-from ./work_dirs/sac_transformer_drawer/env1000/SAC/models/model_2000000.ckpt \
--cfg-options "env_cfg.env_name=OpenCabinetDrawer_1000_link_0-v0" "eval_cfg.num=1000" "eval_cfg.num_procs=1" "eval_cfg.use_log=True" "eval_cfg.save_traj=True"


# python -m tools.run_rl configs/sac/sac_mani_skill_state_1M_train.py --gpu-ids=0 \
# 	--work-dir=./work_dirs/sac_transformer_drawer/env1000/ \
# 	--cfg-options "env_cfg.env_name=OpenCabinetDrawer_1000-v0" "eval_cfg.save_video=True" 


# python -m tools.run_rl configs/sac/sac_mani_skill_state_1M_train.py --gpu-ids=0 \
# 	--work-dir=./work_dirs/sac_transformer_drawer/env1000_link0/ \
# 	--cfg-options "env_cfg.env_name=OpenCabinetDoor_1000_link_0-v0" "eval_cfg.save_video=True" 
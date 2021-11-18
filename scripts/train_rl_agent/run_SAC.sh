# python -m tools.run_rl configs/sac/sac_mani_skill_state_1M_train.py --seed=0 --cfg-options "env_cfg.env_name=OpenCabinetDrawer_1000_link_0-v0" --gpu-ids=0 --clean-up
model_list=$(python -c "import mani_skill, os, os.path as osp; print(osp.abspath(osp.join(osp.dirname(mani_skill.__file__), 'assets', 'config_files', 'cabinet_models_drawer.yml')))")


python -m tools.run_rl configs/sac/sac_mani_skill_state_1M_train.py --gpu-ids=0 \
	--work-dir=./work_dirs/sac_transformer_drawer/ \
	--cfg-options "train_mfrl_cfg.total_steps=5000" "train_mfrl_cfg.n_steps=0" "train_mfrl_cfg.init_replay_buffers=" \
	"train_mfrl_cfg.init_replay_with_split=[\"./full_mani_skill_data/OpenCabinetDrawer/\",\"$model_list\"]" \
	"env_cfg.env_name=OpenCabinetDrawer-v0" "eval_cfg.num=10" "eval_cfg.num_procs=1" "train_mfrl_cfg.n_eval=10" "eval_cfg.save_video=True"
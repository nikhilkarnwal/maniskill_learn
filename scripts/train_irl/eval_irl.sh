python -m tools.run_rl configs/bc/mani_skill_gail_point_cloud_transformer.py --evaluation --gpu-ids=0 \
--work-dir=./test/gail_transformer_drawer/ \
--resume-from ./full_mani_skill_data/models/OpenCabinetDrawer-v0_PN_Transformer.ckpt \
--cfg-options "env_cfg.env_name=OpenCabinetDrawer-v0" "eval_cfg.num=30" "eval_cfg.num_procs=1" "eval_cfg.use_log=True" "eval_cfg.save_traj=False" \
"agent.policy_cfg.gail_cfg.resume=True" "agent.policy_cfg.gail_cfg.save=False"


from copy import copy, deepcopy
from datetime import datetime
import abc
import os
from tabnanny import verbose
import numpy as np
from sqlalchemy import false
from stable_baselines3.common import vec_env
from stable_baselines3.common.base_class import maybe_make_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import torch
import torch.nn.functional as F
from mani_skill_learn.env.env_utils import build_env
from mani_skill_learn.env.replay_buffer import ReplayBufferAS, ReplayBufferFHAS
from mani_skill_learn.env.wrappers import build_wrapper
from mani_skill_learn.methods.builder import BRL

from mani_skill_learn.networks import build_model
from mani_skill_learn.optimizers import build_optimizer
from mani_skill_learn.utils.data import to_torch
from mani_skill_learn.utils.data.converter import to_np
from imitation.util import logger
from mani_skill_learn.utils.torch import BaseAgent
from mani_skill_learn.networks.builder import MODELNETWORKS, build_backbone, build_dense_head
from mani_skill_learn.networks.utils import replace_placeholder_with_args, get_kwargs_from_shape

from stable_baselines3 import PPO, SAC
from imitation.algorithms.adversarial import gail, airl
from imitation.util import util
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from stable_baselines3.common.noise import NormalActionNoise
from imitation.rewards import reward_nets
from torch import nn
@BRL.register_module()
class IRLStateSB(BaseAgent):

    def __init__(self, policy_cfg, obs_shape, action_shape, action_space, env_cfg, batch_size=128):
        self.gail_config = policy_cfg.pop('gail_cfg')
        super().__init__()
        self.env_cfg = env_cfg
        self.gen_algo = None
        model_time = datetime.now()
        self.current_time = model_time.strftime("%d_%m_%Y_%H_%M_%S")
        self.work_dir = f"/media/biswas/D/maniskill/IRLStateSB/{self.gail_config['algo']}/{self.gail_config['gen_algo']}/{self.current_time}/"

    def set_evaluate(self):
        self.gen_algo.evaluate_actions = self.gen_algo.get

    def update_learning_rate(self, optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
        """
        Update the learning rate for a given optimizer.
        Useful when doing linear schedule.

        :param optimizer:
        :param learning_rate:
        """
        for param_group in optimizer.param_groups:
            param_group["lr"] *= learning_rate

    def gail_callable(self,round):
        if self.scheduler != None:
            self.scheduler.step(round)
        # n_round = self.gail_config['total_timesteps'] // self.model.gen_train_timesteps
        # lr = 1 - round/n_round
        # if self.gail_config['lr_update']:
        #     self.update_learning_rate(self.model._disc_opt, lr)

            
    def setup_model(self):
        env = build_env(self.env_cfg)
        self.env = env

        # setting replay buffer
        replay_buffer_kwargs={'ep_max_len': env._max_episode_steps}
        if self.gail_config['finiteH_as'] == 2:
            replay_bf_cls = ReplayBufferFHAS
        elif self.gail_config['finiteH_as'] == 1:
            replay_bf_cls = ReplayBufferAS
        else:
            replay_bf_cls = None
            replay_buffer_kwargs = None

        action_noise = None
        n_actions = env.action_space.shape[-1]
        if self.gail_config.get('explore',false):
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))
        # setting gen algo
        gen_cls = PPO
        if self.gail_config['gen_algo'] == 'ppo':
            self.gen_algo = PPO(env=env, device='cuda', tensorboard_log="IRLStateSB/logs/"+self.current_time, **
                                self.gail_config["ppo_algo"])
        elif self.gail_config['gen_algo'] == 'sac':
            gen_cls = SAC
            self.gen_algo = SAC(env=env, device='cuda', tensorboard_log="IRLStateSB/logs/"+self.current_time, **
                                self.gail_config["sac_algo"], replay_buffer_class=replay_bf_cls,
                                action_noise=action_noise, 
                                replay_buffer_kwargs=replay_buffer_kwargs
                                # ,
                                # policy_kwargs={
                                # "activation_fn":nn.Tanh,
                                # "net_arch":[512,512]}
                                )
        self.gail_config["policy_model"] = self.gail_config["gen_algo"] + \
            "_"+self.gail_config["policy_model"]
        if self.gail_config['resume']:
            print(f'Loading policy from -{self.gail_config["resume_model"]}')
            self.gen_algo = gen_cls.load(self.gail_config['resume_model'], print_system_info=True, env=env)

        
        # self.venv = DummyVecEnv([lambda: env])
        self.venv = self.gen_algo._wrap_env(env)
        reward_net = reward_nets.BasicRewardNet(
                observation_space=self.venv.observation_space,
                action_space=self.venv.action_space
            )
        for key in reward_net.mlp._modules.keys():
            if isinstance(reward_net.mlp._modules[key],nn.Linear):
                reward_net.mlp._modules[key] = nn.utils.spectral_norm(reward_net.mlp._modules[key])
        # reward_net = None
        if self.gail_config['algo'] == 'gail':
            self.model = gail.GAIL(demonstrations=None,
                                   reward_net=reward_net,
                                   venv=self.gen_algo.get_env(), 
                                   custom_logger = logger.configure(self.work_dir),
                                   gen_algo=self.gen_algo, log_dir = self.work_dir,
                                   **self.gail_config["irl_algo"])
        elif self.gail_config['algo'] == 'airl':
            self.model = airl.AIRL(demonstrations=None,
                                   venv=self.gen_algo.get_env(), 
                                   custom_logger = logger.configure(self.work_dir),
                                   gen_algo=self.gen_algo, log_dir = self.work_dir,
                                   **self.gail_config["irl_algo"])

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.model._disc_opt, gamma=self.gail_config["disc_lr_exp"], verbose=True)
        # self.scheduler = None
        #setting callbacks for irl
        self.callbks = []
        self.callbks.append(CheckpointCallback(
                    save_freq=self.gail_config['save_freq'],
                    save_path=self.work_dir,
                    name_prefix="rl_model",
                    verbose=1,
                ))
        self.callbks.append(EvalCallback(
                self.env,
                best_model_save_path=self.work_dir,
                n_eval_episodes=20,
                log_path=self.work_dir,
                eval_freq=self.gail_config['eval_freq']
            ))
        self.model.gen_callback = [*self.callbks,self.model.gen_callback]

        if self.gail_config['true_reward']:
            self.gen_algo.set_env(env)
        self.obs = None

        # if self.gail_config['resume'] and self.gail_config['algo'] == 'gail' and self.gail_config['reward_model']!=None and os.path.exists(self.gail_config['reward_model']):
        #     print(f'Loading reward from -{self.gail_config["reward_model"]}')
        #     self.model._reward_net.load_state_dict(
        #         torch.load(self.gail_config['reward_model']))
            # self.model._reward_net.to(self.device)

    def update_demo(self, demo):
        demo.process()
        sampled_batch = demo.get_all()  # demo.sample(self.batch_size)
        print(f'Demo size-{len(sampled_batch)}')
        self.model.set_demonstrations(sampled_batch)

    def save_model(self):
        print('Saving GAIL models', self.current_time +
                  "_"+self.gail_config['policy_model'])
        self.gen_algo.save(
            self.work_dir+self.gail_config['policy_model'])
        torch.save(self.model._reward_net.state_dict(),
                    self.work_dir+self.gail_config['reward_model'])

    def update_parameters(self, re, **kvargs):
        if self.gen_algo == None:
            self.setup_model()
        self.update_demo(re)
        if self.gail_config['pretrain']:
            print("Pretraining started!")
            curr_env = self.model.gen_algo.get_env()
            curr_learning_start = self.model.gen_algo.learning_starts
            self.model.gen_algo.set_env(self.env)
            # self.model.gen_algo.set_env(vec_env.VecNormalize(self.venv))
            self.model.gen_algo.learning_starts=4000
            self.model.gen_algo.learn(total_timesteps=self.gail_config['pretrain_ts'],callback=self.callbks)
            self.model.gen_algo.learning_starts = curr_learning_start
            self.model.gen_algo.set_env(curr_env)
            print("Pretraining done!")
            print("Running 1 round of disc")

        self.model.train(
            total_timesteps=self.gail_config['irl_timesteps'], callback=self.gail_callable)
        print(f"IRL trained for {self.gail_config['irl_timesteps']}")

        if (self.gail_config['total_timesteps'] - self.gail_config['irl_timesteps']) > 0:
            print(f"Learning Gen algo for {self.gail_config['total_timesteps']-self.gail_config['irl_timesteps']}")
            self.model.train_gen(self.gail_config['total_timesteps']-self.gail_config['irl_timesteps'])

        if self.gail_config['save']:
            self.save_model()
        return {
            'policy_abs_error': 1,
            'policy_loss': 1
        }

    def forward(self, obs, **kwargs):
        mean = np.zeros_like(obs)
        std = np.ones_like(obs)
        if self.gen_algo == None:
            self.setup_model()
        # if self.obs == None:
        #     self.obs = []
        # elif len(self.obs) >= 100:
        #     mean = np.mean(self.obs,axis=0)
        #     std = np.std(self.obs,axis=0)
        #     self.obs.append(obs[0])
        # obs[0] = (obs[0] - mean)/(std+1e-5)
            # print(obs)

        self.gen_algo.env.obs_rms.update(obs)
        obs = self.gen_algo.env.normalize_obs(obs)
        x = self.gen_algo.predict(obs)[0]
        # print(x)
        return x

from imitation.algorithms import bc

@BRL.register_module()
class BCStateSB(IRLStateSB):

    def __init__(self, policy_cfg, obs_shape, action_shape, action_space, env_cfg, batch_size=128):
        super().__init__(policy_cfg, obs_shape, action_shape, action_space, env_cfg, batch_size=batch_size)

    def setup_model(self):
        env = build_env(self.env_cfg)
        self.env = env
        self.model = bc.BC(
            demonstrations=None, 
            observation_space=env.observation_space,
            action_space=env.action_space,
            custom_logger=logger.configure(self.work_dir),
            **self.gail_config['bc_config']['init'])


    def save_model(self):
        self.model.save_policy(f"{self.work_dir}/bc_model.pth")

    def update_parameters(self, re, **kvargs):
        self.setup_model()
        self.update_demo(re)
        self.model.train(**self.gail_config['bc_config']['train'])
        return {
            'policy_abs_error': 1,
            'policy_loss': 1
        }

    def forward(self, obs, **kwargs):
        x = self.model.policy.predict(obs)[0]
        # print(x)
        return x

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
from mani_skill_learn.methods.irl.state_adversarial import IRLStateSB

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
        self.save_model()
        return {
            'policy_abs_error': 1,
            'policy_loss': 1
        }

    def forward(self, obs, **kwargs):
        x = self.model.policy.predict(obs)[0]
        # print(x)
        return x
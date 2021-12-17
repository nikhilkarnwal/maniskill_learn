
from datetime import datetime
import abc
import torch
from torch._C import device
import torch.nn.functional as F
from mani_skill_learn.env.env_utils import build_env
from mani_skill_learn.env.wrappers import build_wrapper
from mani_skill_learn.methods.builder import BRL

from mani_skill_learn.networks import build_model
from mani_skill_learn.optimizers import build_optimizer
from mani_skill_learn.utils.data import to_torch
from mani_skill_learn.utils.data.converter import to_np
from mani_skill_learn.utils.meta import logger
from mani_skill_learn.utils.torch import BaseAgent
from mani_skill_learn.networks.builder import build_backbone, build_dense_head
from mani_skill_learn.networks.utils import replace_placeholder_with_args, get_kwargs_from_shape

from stable_baselines3 import PPO, SAC
from imitation.algorithms.adversarial import gail, airl
from imitation.util import util


@BRL.register_module()
class RLSB(BaseAgent):

    def __init__(self, policy_cfg, obs_shape, action_shape, action_space, env_cfg, batch_size=128):
        self.gail_config = policy_cfg.pop('gail_cfg')
        super().__init__()
        self.env_cfg = env_cfg
        self.gen_algo = None

    def setup_model(self):
        env = build_env(self.env_cfg)
        if self.gail_config['gen_algo'] == 'ppo':
            self.gen_algo = PPO(env=env, device='cuda', tensorboard_log = 'RLSB', **
                                self.gail_config["ppo_algo"])
        elif self.gail_config['gen_algo'] == 'sac':
            self.gen_algo = SAC(env=env, device='cuda', tensorboard_log = 'RLSB', **
                                self.gail_config["sac_algo"])
        self.gail_config["policy_model"] = self.gail_config["gen_algo"] + \
            "_"+self.gail_config["policy_model"]
        if self.gail_config['resume']:
            print(f'Loading policy from -{self.gail_config["resume_model"]}')
            self.gen_algo.load(self.gail_config['resume_model'])

    def update_parameters(self, re, **kvargs):
        if self.gen_algo == None:
            self.setup_model()
        self.gen_algo.learn(
            total_timesteps=self.gail_config['total_timesteps'])

        if self.gail_config['save']:
            now = datetime.now()
            current_time = now.strftime("%d_%m_%Y-%H_%M_%S")
            print('Saving GAIL models', current_time +
                  "_"+self.gail_config['policy_model'])
            self.gen_algo.save(
                current_time+"_"+self.gail_config['policy_model'])
        return {
            'policy_abs_error': 1,
            'policy_loss': 1
        }

    def forward(self, obs, **kwargs):
        if self.gen_algo == None:
            self.setup_model()
        x = self.gen_algo.predict(obs)[0]
        # print(x)
        return x

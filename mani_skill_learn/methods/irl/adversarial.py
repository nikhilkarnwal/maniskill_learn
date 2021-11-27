'''
IRL algorithms
'''

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

from stable_baselines3 import PPO, ppo
from imitation.algorithms.adversarial import gail, airl
from imitation.util import util


class Adversarial(BaseAgent):
    def __init__(self, policy_cfg, obs_shape, action_shape, action_space, batch_size=128):
        super().__init__()

        self.batch_size = batch_size

        policy_optim_cfg = policy_cfg.pop("optim_cfg")
        policy_cfg['obs_shape'] = obs_shape
        policy_cfg['action_shape'] = action_shape
        policy_cfg['action_space'] = action_space

        self.policy = build_model(policy_cfg)
        # # nn_cfg['obs_shape'] = obs_shape
        # # nn_cfg['action_shape'] = action_shape
        # # nn_cfg['action_space'] = action_space

        # replaceable_kwargs = get_kwargs_from_shape(obs_shape, action_shape)
        # nn_cfg = replace_placeholder_with_args(nn_cfg, **replaceable_kwargs)
        # self.backbone = build_backbone(nn_cfg)

    @abc.abstractmethod
    def update_parameters():
        raise NotImplementedError("")


# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from imitation.util import util
# import torch as th
# import torch.nn as nn


# class CustomCombinedExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space, backbone, features_dim):
#         # We do not know features-dim here before going over all the items,
#         # so put something dummy for now. PyTorch requires calling
#         # nn.Module.__init__ before adding modules
#         super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=features_dim)

#         extractors = {}

#         self.backbone = backbone

#         # Update the features dim manually
#         self._features_dim = features_dim

#     def forward(self, observations) -> th.Tensor:


#         return th.cat(encoded_tensor_list, dim=1)

@BRL.register_module()
class GAILSB(Adversarial):

    def __init__(self, policy_cfg, obs_shape, action_shape, action_space, env_cfg, batch_size=128):
        self.gail_config = policy_cfg.pop('gail_cfg')
        super().__init__(policy_cfg, obs_shape, action_shape,
                         action_space, batch_size=batch_size)
        self.env_cfg = env_cfg
        self.gen_algo = None
    
    def setup_model(self):
        env = build_env(self.env_cfg)
        env.device = 'cuda'
        env.set_backbone(self.policy.backbone)
        self.gen_algo = PPO("MlpPolicy", env=env, verbose=True,
                            device='cuda', tensorboard_log='GAILSB', 
                            n_steps=self.gail_config['timesteps'], batch_size=self.gail_config['demo_batch_size'])
        self.gail_config["policy_model"] = self.gail_config["algo"] + \
            "_"+self.gail_config["policy_model"]
        if self.gail_config['resume']:
            print(f'Loading policy from -{self.gail_config["policy_model"]}')
            self.gen_algo.load(self.gail_config['policy_model'])

        if self.gail_config['algo'] == 'gail':
            self.model = gail.GAIL(demonstrations=None, demo_batch_size=self.gail_config['demo_batch_size'],
                                   venv=self.gen_algo.get_env(), gen_algo=self.gen_algo, n_disc_updates_per_round=10)

        # if self.gail_config['resume'] and self.gail_config['algo'] == 'gail':
        #     print(f'Loading reward from -{self.gail_config["reward_model"]}')
        #     self.model._reward_net.load_state_dict(
        #         torch.load(self.gail_config['reward_model'],'cpu'))
        #     self.model._reward_net.to(self.device)

    def update_demo(self, demo):
        demo.process(self.policy.backbone, self.device)
        sampled_batch = demo.get_all()  # demo.sample(self.batch_size)
        print(f'Demo size-{sampled_batch.shape}')
        self.model.set_demonstrations(sampled_batch)

    def update_parameters(self, re, **kvargs):
        if self.gen_algo == None:
            self.setup_model()
        self.policy.backbone.eval()
        if self.gail_config['algo'] == 'gail':
            self.update_demo(re)
            self.model.train(
                total_timesteps=self.gail_config['total_timesteps'])
        else:
            self.gen_algo.learn(
                total_timesteps=self.gail_config['total_timesteps'])

        if self.gail_config['save']:
            print({'Saving GAIL models'})
            self.gen_algo.save(self.gail_config['policy_model'])
            # torch.save(self.model._reward_net.state_dict(),self.gail_config['reward_model'])
        return {
            'policy_abs_error': 1,
            'policy_loss': 1
        }

    def forward(self, obs, **kwargs):
        if self.gen_algo == None:
            self.setup_model()
        self.policy.backbone.eval()
        # print(obs)
        obs = to_torch(obs, device=self.device, dtype='float32')
        # if 'pointcloud' in obs:
        #     curr_obs = obs['pointcloud']
        #     for key in curr_obs:
        #         if not isinstance(curr_obs[key], dict):
        #             curr_obs[key] = curr_obs[key]
        # obs['state']=obs['state']
        # print(obs)
        obs = self.policy.backbone(obs)[1][0]
        obs = to_np(obs)
        # print(obs)
        x = self.gen_algo.predict(obs)[0]
        # print(x)
        return x

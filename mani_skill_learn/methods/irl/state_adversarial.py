
from datetime import datetime
import abc
import torch
import torch.nn.functional as F
from mani_skill_learn.env.env_utils import build_env
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


@BRL.register_module()
class IRLStateSB(BaseAgent):

    def __init__(self, policy_cfg, obs_shape, action_shape, action_space, env_cfg, batch_size=128):
        self.gail_config = policy_cfg.pop('gail_cfg')
        super().__init__()
        self.env_cfg = env_cfg
        self.gen_algo = None
        model_time = datetime.now()
        self.current_time = model_time.strftime("%d_%m_%Y_%H_%M_%S")
        self.work_dir = f"IRLStateSB/{self.gail_config['algo']}/{self.gail_config['gen_algo']}/{self.current_time}/"

    def set_evaluate(self):
        self.gen_algo.evaluate_actions = self.gen_algo.get

    def setup_model(self):
        env = build_env(self.env_cfg)
        self.env = env
        if self.gail_config['gen_algo'] == 'ppo':
            self.gen_algo = PPO(env=env, device='cuda', tensorboard_log="IRLStateSB/logs/"+self.current_time, **
                                self.gail_config["ppo_algo"])
        elif self.gail_config['gen_algo'] == 'sac':
            self.gen_algo = SAC(env=env, device='cuda', tensorboard_log="IRLStateSB/logs/"+self.current_time, **
                                self.gail_config["sac_algo"])
        self.gail_config["policy_model"] = self.gail_config["gen_algo"] + \
            "_"+self.gail_config["policy_model"]
        if self.gail_config['resume']:
            print(f'Loading policy from -{self.gail_config["resume_model"]}')
            self.gen_algo.load(self.gail_config['resume_model'])

        if self.gail_config['algo'] == 'gail':
            self.model = gail.GAIL(demonstrations=None,
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

        if self.gail_config['true_reward']:
            self.gen_algo.set_env(env)

        if self.gail_config['resume'] and self.gail_config['algo'] == 'gail':
            print(f'Loading reward from -{self.gail_config["reward_model"]}')
            self.model._reward_net.load_state_dict(
                torch.load(self.gail_config['reward_model'], 'cpu'))
            self.model._reward_net.to(self.device)

    def update_demo(self, demo):
        demo.process()
        sampled_batch = demo.get_all()  # demo.sample(self.batch_size)
        print(f'Demo size-{sampled_batch.shape}')
        self.model.set_demonstrations(sampled_batch)

    def update_parameters(self, re, **kvargs):
        if self.gen_algo == None:
            self.setup_model()
        self.update_demo(re)
        if self.gail_config['pretrain']:
            print("Pretraining started!")
            curr_env = self.model.gen_algo.get_env()
            self.gen_algo.set_env(self.env)
            self.gen_algo.learn(total_timesteps=self.gail_config['total_timesteps']//100)
            self.gen_algo.set_env(curr_env)
            print("Pretraining done!")
        self.model.train(
            total_timesteps=self.gail_config['total_timesteps'])
        print(self.model.gen_train_timesteps)

        if self.gail_config['save']:
            print('Saving GAIL models', self.current_time +
                  "_"+self.gail_config['policy_model'])
            self.gen_algo.save(
                self.work_dir+self.gail_config['policy_model'])
            torch.save(self.model._reward_net.state_dict(),
                       self.work_dir+self.gail_config['reward_model'])
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

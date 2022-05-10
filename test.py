from typing import Callable
from hashlib import new
from botocore import vendored
import gym
from gym.spaces import space
import mani_skill.env
from numpy.core.fromnumeric import shape


env = gym.make('OpenCabinetDoorReach-v0')
# full environment list can be found in available_environments.txt

env.set_env_mode(obs_mode='state', reward_type='dense')
# obs_mode can be 'state', 'pointcloud' or 'rgbd'
# reward_type can be 'sparse' or 'dense'
print(env.observation_space) # this shows the observation structure in Openai Gym's format
print(env.action_space) # this shows the action space in Openai Gym's format
obs = env.reset()
# env.render('human')
print(obs)
for level_idx in range(0, 5): # level_idx is a random seed
    obs = env.reset(level=level_idx)
    # print('#### Level {:d}'.format(level_idx))
    for i_step in range(10000):
        # env.render('human') # a display is required to use this function; note that rendering will slow down the running speed
        action = env.action_space.sample()
        env.render("human")
        obs, reward, done, info = env.step(action) # take a random action
        print('{:d}: reward {:.4f}, done {}, info {}'.format(i_step, reward, done,info))
        if done:
            break
env.close()


# from h5py import File

# from mani_skill_learn.utils.fileio.h5_utils import load_h5_as_dict_array, load_h5s_as_list_dict_array
# f = File('./full_mani_skill_state_data/OpenCabinetDrawer_state/OpenCabinetDrawer_1000_link_0-v0.h5', 'r')
# # f is a h5py.Group with keys traj_0 ... traj_n
# print(f['traj_0'].keys())
# print(load_h5_as_dict_array(f['traj_0'])['obs'].shape)

# print(load_h5_as_dict_array(f['traj_0']['obs']).shape)


# import gym
# import torch as th
# from torch import nn

# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from imitation.util import util


# class CustomCombinedExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Dict):
#         # We do not know features-dim here before going over all the items,
#         # so put something dummy for now. PyTorch requires calling
#         # nn.Module.__init__ before adding modules
#         super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

#         extractors = {}

#         total_concat_size = 0
#         # We need to know size of the output of this extractor,
#         # so go over all the spaces and compute output feature sizes
#         for key, subspace in observation_space.spaces.items():
#             if key == "image":
#                 # We will just downsample one channel of the image by 4x4 and flatten.
#                 # Assume the image is single-channel (subspace.shape[0] == 0)
#                 extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
#                 total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
#             elif key == "agent":
#                 # Run through a simple MLP
#                 extractors[key] = nn.Linear(subspace.shape[0], 16)
#                 total_concat_size += 16

#         self.extractors = nn.ModuleDict(extractors)

#         # Update the features dim manually
#         self._features_dim = total_concat_size

#     def forward(self, observations) -> th.Tensor:
#         encoded_tensor_list = []

#         # self.extractors contain nn.Modules that do all the processing.
#         for key, extractor in self.extractors.items():
#             encoded_tensor_list.append(extractor(observations[key]))
#         # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
#         return th.cat(encoded_tensor_list, dim=1)

# from stable_baselines3 import PPO

# policy_kwargs = dict(
#     features_extractor_class=CustomCombinedExtractor,
#     # features_extractor_kwargs=dict(features_dim=128),
# )

# def set_mode(env, i):
#     env = gym.make('OpenCabinetDrawer-v0')
#     env.set_env_mode(obs_mode='state', reward_type='sparse')
#     print('Called')
#     # print(env.observation_space)
#     return env

# # from stable_baselines3.common.vec_env import vec_extract_dict_obs, vec_monitor
# # vec_extract_dict_obs.VecExtractDictObs


# import numpy as np

# class NewEnv(gym.ObservationWrapper):
#     def __init__(self, env: gym.Env, i) -> None:
#         super().__init__(env)
#         # new_obs_space = self.observation_space['pointcloud']
#         # new_obs_space['agent']=gym.spaces.Box(-np.inf,np.inf,shape=self.observation_space['agent'],dtype=np.float32)
#         if isinstance(self.observation_space,dict):
#             self.observation_space = gym.spaces.Dict(self.flatten_dict(self.observation_space))
#         else:
#             self.observation_space = gym.spaces.Dict({'agent':self.observation_space})
#         print(self.observation_space)

#     def flatten_dict(self, obs):
#         new_obs = {}
#         for key in obs.keys():
#             if isinstance(obs[key],dict):
#                 new_obs.update(obs[key])
#         if 'agent' in obs.keys():
#             new_obs['agent']=gym.spaces.Box(-np.inf,np.inf,shape=obs['agent'],dtype=np.float32)
#         return new_obs

#     def observation(self, obs):
#         if isinstance(obs, dict):
#             new_obs = {}
#             for key in obs.keys():
#                 if isinstance(obs[key],dict):
#                     new_obs.update(obs[key])
#             if 'agent' in obs.keys():
#                 new_obs['agent']=obs['agent']
#             obs = new_obs
#         else:
#             obs = {'agent':obs}
#         return obs

# def func1(temp:gym.spaces.Space):
#     print(temp.spaces)
#     return temp
# temp = func1(env.observation_space)
# print(type(env.observation_space))
# venv = util.make_vec_env('OpenCabinetDrawer-v0', n_envs=1, parallel=False, post_wrappers=[set_mode, NewEnv])
# # venv = vec_extract_dict_obs.VecExtractDictObs(venv)
# venv = gym.make('OpenCabinetDrawer-v0')
# venv = set_mode(venv,0)
# venv = NewEnv(venv,0)
# model = PPO('MultiInputPolicy', env = venv, policy_kwargs=policy_kwargs, verbose=True)
# print(model.policy)
# model.learn(100)

# from h5py import File

# from mani_skill_learn.utils.fileio.h5_utils import load_h5_as_dict_array
# f = File('./full_mani_skill_data/OpenCabinetDrawer/OpenCabinetDrawer_1056_link_0-v0.h5', 'r')
# # f is a h5py.Group with keys traj_0 ... traj_n
# print(f['traj_0'].keys())
# print(load_h5_as_dict_array(f['traj_0']))

# import torch
# PATH = './full_mani_skill_data/models/OpenCabinetDrawer-v0_PN_Transformer.ckpt'
# x = torch.load(PATH)
# print(x)

"""Trains BC, GAIL and AIRL models on saved CartPole-v1 demonstrations."""

import argparse
import os
import numpy as np
import gym
from imitation.data.types import TrajectoryWithRew, save, load
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from datetime import datetime
import pathlib
import pickle
import tempfile

import stable_baselines3 as sb3

from imitation.algorithms import bc
from imitation.algorithms.adversarial import airl, gail
from imitation.data import rollout
from imitation.util import logger, util
import stable_baselines3.common.utils as s3utils
from stable_baselines3.sac.sac import SAC
# from mani_skill_learn import env

# from mani_skill_learn.env.replay_buffer import ReplayBufferFHAS, TrajReplayStateABS



def run_gail(name, transitions, args, work_dir):
    # # Load pickled test demonstrations.
    # with open("./final.pkl", "rb") as f:
    #     # This is a list of `imitation.data.types.Trajectory`, where
    #     # every instance contains observations and actions for a single expert
    #     # demonstration.
    #     trajectories = pickle.load(f)

    # # Convert List[types.Trajectory] to an instance of `imitation.data.types.Transitions`.
    # # This is a more general dataclass containing unordered
    # # (observation, actions, next_observation) transitions.
    # transitions = rollout.flatten_trajectories(trajectories)
    print(transitions[0])

    venv = util.make_vec_env(name, n_envs=1)

    tempdir_path = pathlib.Path(work_dir)
    print(
        f"All Tensorboards and logging are being written inside {tempdir_path}/.")

    # Train BC on expert data.
    # BC also accepts as `demonstrations` any PyTorch-style DataLoader that iterates over
    # dictionaries containing observations and actions.
    bc_logger = logger.configure(tempdir_path / "BC/")
    # bc_trainer = bc.BC(
    #     observation_space=venv.observation_space,
    #     action_space=venv.action_space,
    #     demonstrations=transitions,
    #     custom_logger=bc_logger,
    # )
    # bc_trainer.train(n_epochs=1)

    # Train GAIL on expert data.
    # GAIL, and AIRL also accept as `demonstrations` any Pytorch-style DataLoader that
    # iterates over dictionaries containing observations, actions, and next_observations.
    gail_logger = logger.configure(tempdir_path / "GAIL/")
    if args.gen == 'ppo':
        gen_algo = sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=2048)
        gen_algo.policy
    else:
        sac_algo = dict(
            buffer_size=1000000,
            learning_rate=0.0003,
            learning_starts=500,
            batch_size=1024,
            gamma=0.95,
            verbose=1,
            seed=101,
            policy='MlpPolicy'
        )
        gen_algo = sb3.SAC(env=venv, **sac_algo)
        gen_algo.policy.evaluate_actions = gen_algo.policy.actor.action_dist.log_prob()
    # gen_algo.learn(100000)
    # gen_algo.collect_rollouts()

    gail_trainer = airl.AIRL(
        venv=venv,
        demonstrations=transitions,
        demo_batch_size=32,
        gen_algo=gen_algo,
        custom_logger=gail_logger, n_disc_updates_per_round=2, normalize_reward=True, normalize_obs=True,
        init_tensorboard_graph=True,
        allow_variable_horizon=True,gen_train_timesteps = 1*1024,
        # disc_opt_kwargs={'lr': 3e-5},
    )
    
    # gail_trainer = gail.GAIL(
    #     venv=venv,
    #     demonstrations=transitions,
    #     demo_batch_size=32,
    #     gen_algo=gen_algo,
    #     custom_logger=gail_logger, n_disc_updates_per_round=2, normalize_reward=True, normalize_obs=True,
    #     init_tensorboard_graph=True,
    #     allow_variable_horizon=True,gen_train_timesteps = 1*1024,
    #     # disc_opt_kwargs={'lr': 3e-5},
    # )
    gail_trainer.allow_variable_horizon = True
    gail_trainer.train(total_timesteps=1024)

    # Train AIRL on expert data.
    # airl_logger = logger.configure(tempdir_path / "AIRL/")
    # airl_trainer = airl.AIRL(
    #     venv=venv,
    #     demonstrations=transitions,
    #     demo_batch_size=32,
    #     gen_algo=gen_algo,
    #     custom_logger=airl_logger,
    # )
    # airl_trainer.allow_variable_horizon=True
    # airl_trainer.train(total_timesteps=2048*50)

    obs = venv.reset()
    while True:
        action, _states = gen_algo.predict(obs)
        obs, rewards, dones, info = venv.step(action)
        venv.render()


# gen_algo=sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=1024)
# gen_algo.learn(2048)
# venv = gen_algo.env
# obs = venv.reset()
# for i in range(100):
#     obs_tensor = obs_as_tensor(obs,'cuda')
#     actions, values, log_probs = gen_algo.policy.forward(obs_tensor)
#     a = actions.cpu().numpy()
#     obs, rew, done, info = venv.step(a)
#     print(f'obs{obs}, rew{rew}, done{done}, info{info}')
#     gen_algo.ep_info_buffer.clear()
#     gen_algo._update_info_buffer(info)
# print(gen_algo.ep_info_buffer)

def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def run_gen(name,args , work_dir="temp"):
    # Parallel environments
    env = make_vec_env(name, n_envs=1)

    if args.gen == 'ppo':
        model = PPO("MlpPolicy", env, verbose=1,
                    learning_rate=linear_schedule(0.003),tensorboard_log=work_dir)
    else:
        sac_algo = dict(
            buffer_size=1000000,
            learning_rate=0.0003,
            learning_starts=4000,
            batch_size=1024*10,
            gamma=0.95,
            verbose=1,
            seed=101,
            policy='MlpPolicy'
        )
        model = sb3.SAC(env=env,tensorboard_log=work_dir, **sac_algo)
    model.learn(total_timesteps=1024)
    model.save(f"{work_dir}/model")
    return model


def get_traj(name, model, num, render=5):
    trajs = []
    traj_len = []
    env = gym.make(name)
    # Parallel environments
    venv = make_vec_env(name, n_envs=1)
    for i in range(num):
        obs_list = []
        rew_list = []
        actions_list = []
        obs = venv.reset()
        cnt = 0
        while cnt < env.spec.max_episode_steps:
            action, _states = model.predict(obs)
            obs_list.append(obs[0])
            actions_list.append(action[0])
            if i < render:
                venv.render()
            obs, rewards, dones, info = venv.step(action)
            rew_list.append(rewards[0])
            cnt += 1
            if dones[0]:
                # print(cnt)
                break
        obs_list.append(obs[0])
        term = True
        traj_len.append(cnt)
        if cnt == env.spec.max_episode_steps:
            term = False
        trajs.append(TrajectoryWithRew(obs=np.array(obs_list), acts=np.array(actions_list),
                                       rews=np.reshape(rew_list, [len(rew_list), ]), infos=None, terminal=True))
    venv.close()
    print(f'Mean len-{np.mean(traj_len)}')
    return trajs


def test_traj(trajs):
    traj_len = []
    traj_rew = []
    for traj in trajs:
        traj_len.append(traj.obs.shape[0])
        traj_rew.append(traj.rews.shape[0])
    print(f"Testing traj: Len - {np.mean(traj_len)}, Rew - {np.mean(traj_rew)}")

def main():
    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")

    parser = argparse.ArgumentParser(description='Run RL training code')
    # Configurations
    parser.add_argument('--gen', help='gn algo', type=str, default='ppo')
    parser.add_argument('--irl', action='store_true', default=False)
    parser.add_argument('--env', type=str, default="Walker2d-v2")
    parser.add_argument('--trajs', type=str, default=None)
    parser.add_argument('--gen_trajs', action='store_true', default=False)
    parser.add_argument('--test_trajs', action='store_true', default=False)
    parser.add_argument('--num_trajs', help='num of traj to gen',
                        type=int, default=1000)
    args = parser.parse_args()
    name = args.env

    # name = "CartPole-v1"
    work_dir = f"test_env/{name}/{dt_string}/"
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
        print(f"Creating dir-{work_dir}")
    print(f"Storing at {work_dir}")
    with open(f'{work_dir}/desc_file.txt', 'w') as fd:
        fd.write(args.__str__())

    if args.trajs == None:
        args.trajs = 'trajs'

    if args.gen_trajs:
        model = run_gen(name, args, work_dir)
        trajs = get_traj(name, model, args.num_trajs, 0)
        save(f"{work_dir}/{args.trajs}", trajs)
        print("Saved trajs")
    else:
        trajs = load(args.trajs)

    if args.test_trajs:
        test_traj(trajs)
        return 

    if args.irl:
        run_gail(name, trajs, args, work_dir)

def run_trajs():
    import gym
    import mani_skill.env
    from h5py import File
    import time

    f = File('./full_mani_skill_state_data/OpenCabinetDrawer_state/custom_1000_link0_v0.h5', 'r')
    env = gym.make('OpenCabinetDrawer_1000_link_0-v0')
    # # full environment list can be found in available_environments.txt

    env.set_env_mode(obs_mode='state', reward_type='dense')
    # # obs_mode can be 'state', 'pointcloud' or 'rgbd'
    # # reward_type can be 'sparse' or 'dense'
    # print(env.observation_space) # this shows the structure of the observation, openai gym's format
    # print(env.action_space) # this shows the action space, openai gym's format

    traj = "traj_" +str(100)
    actions = f[traj]['actions'][:]
    env_states = f[traj]['env_states'][:]
    env_levels = f[traj]['env_levels'][:]
    obs_list = f[traj]['obs'][:]
    rewards_list = f[traj]['rewards'][:]
    obs = env.reset(level=env_levels[0])

    for i_step in range(len(actions)):
        # env.set_state(env_states[i_step])
        env.render('human') # a display is required to use this function, rendering will slower the running speed
        action = actions[i_step] #env.action_space.sample()
        obs, reward, done, info = env.step(action) # take a random action
        if round(reward,2) != round(rewards_list[i_step],2):
            print("WARN: Different reward: \n"+"got: "+ str(reward)+"\t Expected: "+str(rewards_list[i_step]))
        print('{:d}: reward {:.4f}, done {}'.format(i_step, reward, done))
        if done:
            break
        time.sleep(0.5)
    env.close()

# main()
# run_trajs()

# from mani_skill_learn.env.env_utils import build_env
# if __name__ == "__main__":
#     cfg = Config.fromfile("configs/v2/irl_s3_state_as.py")
#     # print(cfg['env_cfg'])
#     env_cfg = cfg['env_cfg']
#     env = build_env(env_cfg)
#     obs = env.reset()
#     print(obs.shape)
#     print(env.step(np.zeros(13))[0])


# sac_algo = dict(
#         buffer_size=1000000,
#         learning_rate=0.0003,
#         learning_starts=4000,
#         batch_size=1024,
#         gamma=0.95,
#         verbose=1,
#         seed=101,
#         policy='MlpPolicy'
#     )
#     gen_algo = sb3.SAC(env=venv, **sac_algo)
# # gen_algo.learn(100000)
# # gen_algo.collect_rollouts()
# gail_trainer = gail.GAIL(
#     venv=venv,
#     demonstrations=None,
#     demo_batch_size=32,
#     gen_algo=gen_algo,
#     custom_logger='../temp', n_disc_updates_per_round=2, normalize_reward=True, normalize_obs=True,
#     init_tensorboard_graph=True,
#     allow_variable_horizon=True,
#     disc_opt_kwargs={'lr': 3e-5},
# )

def check_replayb_FHAS():
    import gym
    import mani_skill.env
    env = gym.make('Pendulum-v0')
    # # full environment list can be found in available_environments.txt

    gen_algo = sb3.SAC(
        'MlpPolicy',env=env,
        replay_buffer_class=ReplayBufferFHAS,
        replay_buffer_kwargs={'ep_max_len': env._max_episode_steps})
    obs = env.reset()
    obs = np.expand_dims(obs,axis=0)
    gen_algo.replay_buffer.add(obs,obs,np.array([1]),np.array([1]),np.array([1]),[{}])
    print(obs)
    print(gen_algo.replay_buffer.observations[:200])
    print(gen_algo.replay_buffer.dones[:200])
    print(env._max_episode_steps)

# check_replayb_FHAS()


def check_traj_replay():
    from h5py import File
    traj_replay  = TrajReplayStateABS(100,200)
    f = File('./full_mani_skill_state_data/OpenCabinetDrawer_state/custom_1000_link0_v0.h5', 'r')

    # traj = "traj_" +str(100)
    # actions = f[traj]['actions'][:]
    # env_states = f[traj]['env_states'][:]
    # env_levels = f[traj]['env_levels'][:]
    # obs_list = f[traj]['obs'][:]
    # rewards_list = f[traj]['rewards'][:]
    traj_replay.restore('./full_mani_skill_state_data/OpenCabinetDrawer_state/custom_1000_link0_v0.h5',1,4)
    traj_replay.process()
    print(traj_replay.get_all().obs)
    print(traj_replay.get_all().dones[30:50])

# check_traj_replay()
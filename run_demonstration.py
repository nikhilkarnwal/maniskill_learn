import gym
import mani_skill.env
from h5py import File

f = File('./full_mani_skill_state_data/OpenCabinetDrawer/OpenCabinetDrawer_1000_link_0-v0.h5', 'r')
env = gym.make('OpenCabinetDrawer_1000_link_0-v0')
# # full environment list can be found in available_environments.txt

env.set_env_mode(obs_mode='state', reward_type='dense')
# # obs_mode can be 'state', 'pointcloud' or 'rgbd'
# # reward_type can be 'sparse' or 'dense'
# print(env.observation_space) # this shows the structure of the observation, openai gym's format
# print(env.action_space) # this shows the action space, openai gym's format

traj = "traj_" +str(3)
actions = f[traj]['actions'][:]
env_states = f[traj]['env_states'][:]
env_levels = f[traj]['env_levels'][:]
obs_list = f[traj]['obs'][:]
rewards_list = f[traj]['rewards'][:]
obs = env.reset(level=env_levels[0])

for i_step in range(len(actions)):
    env.set_state(env_states[i_step])
    env.render('human') # a display is required to use this function, rendering will slower the running speed
    action = actions[i_step] #env.action_space.sample()
    obs, reward, done, info = env.step(action) # take a random action
    if round(reward,2) != round(rewards_list[i_step],2):
        print("WARN: Different reward: \n"+"got: "+ str(reward)+"\t Expected: "+str(rewards_list[i_step]))
    print('{:d}: reward {:.4f}, done {}'.format(i_step, reward, done))
    if done:
        break
env.close()


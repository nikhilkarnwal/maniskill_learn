from types import new_class
import numpy as np

def add_absorbing_state(obs, horizon=2):
    obs_shape = list(obs.shape)
    new_obs_shape = obs_shape.copy()
    #expand horizon and one dim for absorbing condition
    new_obs_shape[0]=horizon
    new_obs_shape[-1]+=1
    final_obs = np.zeros(new_obs_shape)

    # copy original
    final_obs[:obs_shape[0],:obs_shape[1]] = obs

    # put 1 in absorbing state dim, 0 for other 
    final_obs[:,-1]=1
    final_obs[:obs_shape[0],-1]=0

    return final_obs

def add_action_for_absorbing_states(actions, horizon=1):
    new_shape = list(actions.shape)
    new_shape[0]=horizon
    final_actions = np.zeros(new_shape)
    final_actions[:actions.shape[0],:] = actions
    return final_actions

def test_add_absorbing_state():
    obs = np.array([[1,2,3],[2,3,4]])
    expected_ret =  np.array([[1,2,3,0],[2,3,4,0],[0,0,0,1],[0,0,0,1]])
    actual_ret = add_absorbing_state(obs,4)

    # print(expected_ret.shape, actual_ret.shape)
    assert (expected_ret.shape == actual_ret.shape)
    assert (expected_ret[:,-1] == actual_ret[:,-1]).all()
    assert (expected_ret == actual_ret).all()

def test_add_action_for_absorbing_states():
    actions = np.array([[1,2,3],[2,3,3]])
    expected_ret = np.array([[1,2,3],[2,3,3],[0,0,0]])
    assert (expected_ret == add_action_for_absorbing_states(actions,3)).all() 

if __name__ == "__main__":
    print(f"running {2} tests")
    test_add_absorbing_state()
    print(f"test {1} pass")
    test_add_action_for_absorbing_states()
    print(f"test {2} pass")
import sys
sys.path.insert(1, './utils')  

import gym
from gym.spaces import Discrete, MultiDiscrete, Tuple
import numpy as np
from mujoco_worldgen.util.rotation import mat2quat
from util_w import update_obs_space
from copy import deepcopy
from itertools import compress
from vision import insight, in_cone2d
from mujoco_worldgen.util.sim_funcs import qpos_idxs_from_joint_prefix, qvel_idxs_from_joint_prefix, joint_qvel_idxs, joint_qpos_idxs, body_names_from_joint_prefix

class SimpleWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        # Reset obs space
        self.observation_space = update_obs_space(self.env, {'target_pos': (1, 1)})


    def reset(self):
        obs = self.env.reset()
        sim = self.unwrapped.sim

        self.random_pos = np.array([[np.random.uniform(low=-5.0, high=5.0)]])

        return self.observation(obs)

    def observation(self, obs):
        obs['target_pos'] = self.random_pos
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        simple_rew = np.array([0.0 for a in range(1)])

        sim = self.unwrapped.sim
        agent_body_id = joint_qpos_idxs(sim, f"agent0:tx")
        agent_body = sim.data.qpos[agent_body_id]

        if abs(agent_body[0] - self.random_pos) <= 0.1:
            done = True
        else:
            simple_rew[0] += -abs(agent_body[0] - self.random_pos)
        
        rew += simple_rew
        #print(rew)
        return self.observation(obs), rew, done, info

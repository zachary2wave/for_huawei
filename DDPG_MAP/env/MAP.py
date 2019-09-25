import gym
from gym import spaces
import numpy as np
import env.Scenario as Env
from gym.utils import seeding

class MapEnv(gym.Env):

    def __init__(self):
        self.Num_AP = 9
        self.Num_UE = 100
        self.env = Env.scenario(self.Num_AP, self.Num_UE, 2, 1)
        self.action_space = spaces.Box(low=0, high=1, shape=(2*self.Num_AP,), dtype=np.float32)
        # self.action_space_C = spaces.Box(low=0, high=1, shape=(self.Num_AP,), dtype=np.float32)
        # self.action_space = np.array([self.action_space_P, self.action_space_C])
        # self.observation_space_P = spaces.Box(low=0, high=1, shape=(self.Num_AP,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2*self.Num_AP,), dtype=np.float32)
        # self.observation_space = np.array([self.observation_space_P, self.observation_space_C])

        # 此条件下  最大66dB信噪比
        self.Pmax = 20
        self.Cmax = 66

    def reset(self):
        self.state = np.array(self.Num_AP * 2 * [0.5])
        # print(self.state.shape)
        return self.state

    def step(self, u):
        C = u[0:self.Num_AP]*self.Cmax
        P = u[self.Num_AP:2*self.Num_AP]*self.Pmax
        through = self.env.through_out(P, C)
        reward = through
        s_ = u
        return s_, reward/1000, False, {}
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def render(self, mode='human'):
        # tho.showplot(self.placeAP, self.placeUE, self.state, self.channel, self.connection)
        return {}
if __name__ == '__main__':
    env = MapEnv()
    print(env.action_space.shape)
    print(env.observation_space.shape)
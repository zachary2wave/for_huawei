import pandas as pd
import numpy as np
import gym

pendulum = pd.read_csv("./env/monitorChange.csv")
print(pendulum.shape[0])
[theta, thetadot] = [pendulum["obnow1"][1], pendulum["obnow2"][1]]
print([theta, thetadot])
obs1 = np.array([np.cos(theta), np.sin(theta), thetadot])
print(obs1)
print(obs1.shape)
print("---------------------")
obs2 = np.array([pendulum["obnext1"][0], pendulum["obnext2"][0], pendulum["obnext3"][0]])
print(obs2)
print(obs2.shape)


print("env------------------")
ENV_NAME = 'Pendulum-v0'
env = gym.make(ENV_NAME)
print(env.observation_space.shape)
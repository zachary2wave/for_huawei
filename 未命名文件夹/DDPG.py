import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym


from baselines.ddpg.ddpg_learner import DDPG
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.common.tf_util import load_variables, save_variables
import baselines.common.tf_util as U



dataPrimary = pd.read_csv("data_p/monitorChange.csv")
dataProgress = pd.read_csv("data_p/progress.csv")
action_shape = (1,)
nb_action  = 1
observation_shape = (3,)
t_train_time = 10000
t_test_time = 10000
network = 'mlp'
action_noise = None
param_noise = None
popart=False,
load_path = 'ddpg_model'
load_path = None



memory = Memory(limit=int(1e6), action_shape=action_shape, observation_shape=observation_shape)
critic = Critic(network=network)
actor = Actor(nb_action , network=network)

agent = DDPG(actor, critic, memory, observation_shape, action_shape,
             gamma=0.99, tau=0.01, normalize_returns=False, normalize_observations=True,
             batch_size=32, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=1e-2,
             actor_lr=1e-4, critic_lr=1e-3, enable_popart=popart, clip_norm=None,
             reward_scale=1)
sess = U.get_session()
agent.initialize(sess)
sess.graph.finalize()
agent.reset()


if load_path is None:
    epoch_actor_losses = []
    epoch_critic_losses = []
    # reading date
    for index, row in dataPrimary.iterrows():
        if index > 40000 and index<50000:
            obs = np.array([[np.cos(row.obnow1), np.sin(row.obnow1), row.obnow2]])
            action = np.array([[row.action]])
            r = np.array([[row.reward]])
            new_obs = np.array([[row.obnext1, row.obnext2, row.obnext3]])
            agent.store_transition(obs, action, r, new_obs, np.zeros_like(r))
    # training
    for t_train in range(t_train_time):
        cl, al = agent.train()
        epoch_critic_losses.append(cl)
        epoch_actor_losses.append(al)
        print('step'+str(t_train)+',critic_loss:'+str(cl)+',action_loss:',str(al))
    save_variables('ddpg_model')
else:
    load_variables(load_path)


# plt.figure(1)
# plt.plot(epoch_critic_losses)
# plt.plot(epoch_actor_losses)
# plt.show()

env = gym.make('Pendulum-v0')
obs = env.reset()
for time in range(t_test_time):
    action, q, _, _ = agent.step(obs, apply_noise=False, compute_Q=True)
    s_, r, done, _ = env.step(action)
    obs = s_
    env.render()
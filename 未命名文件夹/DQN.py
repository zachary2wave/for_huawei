import gym
import pandas as pd
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.deepq.models import build_q_func


t_train_time = 2e5
t_test_time = 10000


env = gym.make('CartPole-v0')
action_shape = (1,)
nb_action  = 1
observation_shape = (3,)
dataPrimary = pd.read_csv("data_c/Cartpole-v0.csv",header=1 )
load_path = 'ddpg_model'
load_path = None



q_func = build_q_func('mlp')
act, train, update_target, debug = deepq.build_train(
    make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
    q_func = q_func,
    num_actions=nb_action,
    optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
)
replay_buffer = ReplayBuffer(50000)
U.initialize()
update_target()
episode_rewards = [0.0]


if load_path is None:
    for index, row in dataPrimary.iterrows():
        if index > 2:
            obs = np.array(list(row.obnow))
            action = row.action
            rew = row.reward
            new_obs = [row.obnext1, row.obnext2, row.obnext3]
            replay_buffer.add(obs, action, rew, new_obs, float(done))

    record=[]
    for t in range(t_train_time):
        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
        [td_error,loss] = train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
        if t % 1000 == 0:
            update_target()
        print(loss)
        record.append(loss)
    save_variables('ddpg_model')
else:
    load_variables(load_path)




obs = env.reset()
for t in range(t_test_time):
    action = act(obs[None], stochastic=False, update_eps=-1.0)[0]
    s_, r, done, _ = env.step(action)
    obs=s_
    env.render()

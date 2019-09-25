import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

# from env.MAP import MapEnv
import gym

from rl.processors import WhiteningNormalizerProcessor
from rl.ddpg import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

# MuJoCo是一个用于机器人仿真的物理引擎强化学习工具
class MujocoProcessor(WhiteningNormalizerProcessor):
    def process_action(self, action):
        return np.clip(action, -1., 1.)

# Get the environment and extract the number of actions.
import pandas as pd
pendulum = pd.read_csv("./env/monitorChange.csv")

ENV_NAME = 'Pendulum-v0'
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

'''
env = MapEnv()
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]
'''

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
'''
actor.add(Dense(64))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('tanh'))
'''

actor.add(Dense(30))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('tanh'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
'''
x = Dense(400)(flattened_observation)
x = Activation('relu')(x)
x = Concatenate()([x, action_input])
x = Dense(300)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
# x = Activation('linear')(x)
x = Activation('linear')(x)
'''


x = Concatenate()([flattened_observation, action_input])
x = Dense(30)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.1)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                  random_process=random_process, gamma=0.9, target_model_update=1000)
# 发散的可能原因：学习率太大
agent.compile([Adam(lr=1e-4), Adam(lr=1e-4)], metrics=['mae'])


# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C


from collections import namedtuple
# experiencesss = []
Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')
# 这个循环是把CSV文件的数据存进经验池
for  i in range(pendulum.shape[0]):
    if i > 40000:
        theta, thetadot = pendulum["obnow1"][i], pendulum["obnow2"][i]
        state0 = np.array([np.cos(theta), np.sin(theta), thetadot])
        action = pendulum['action'][i]
        reward = pendulum['reward'][i]
    # state1 = np.array([pendulum["obnext1"][1], pendulum["obnext2"][1], pendulum["obnext3"][1]])
        terminal1 = False
        agent.memory.append(state0, action, reward, terminal1, training=True)
    # 这里只存state0就好，不用存state1，因为它sample的时候state0是observations[idx - 1]，state1是[np.copy(x) for x in state0[1:]]
    # experiencesss.append(Experience(state0=state0, action=action, reward=reward, state1=state1, terminal1=terminal1))


from random import sample
# （backward）这个循环是用经验池里的数据反向传播训练actor和critic网络，好像是只训练了actor，critic实在compile函数里训练的
nb_steps = 200000
metricsss = []
for i in range(nb_steps):
    # experiences = sample(experiencesss, agent.batch_size)
    experiences = agent.memory.sample(agent.batch_size)
    assert len(experiences) == agent.batch_size

    # Start by extracting the necessary parameters (we use a vectorized implementation).
    state0_batch = []
    reward_batch = []
    action_batch = []
    terminal1_batch = []
    state1_batch = []
    for e in experiences:
        state0_batch.append(e.state0)
        state1_batch.append(e.state1)
        reward_batch.append(e.reward)
        # action_batch需要的是一个二维数据，如果action_batch.append(e.action)的话就只是一个list
        action_batch.append(e.action)
        terminal1_batch.append(0. if e.terminal1 else 1.)

    # Prepare and validate parameters.
    state0_batch = agent.process_state_batch(state0_batch)
    state1_batch = agent.process_state_batch(state1_batch)
    terminal1_batch = np.array(terminal1_batch)
    reward_batch = np.array(reward_batch)
    action_batch = np.array(action_batch)
    assert reward_batch.shape == (agent.batch_size,)
    assert terminal1_batch.shape == reward_batch.shape
    assert action_batch.shape == (agent.batch_size, agent.nb_actions)
    # assert action_batch.shape == (agent.batch_size,)

    # Update critic.
    if i > agent.nb_steps_warmup_critic:
        target_actions = agent.target_actor.predict_on_batch(state1_batch)
        assert target_actions.shape == (agent.batch_size, agent.nb_actions)
        if len(agent.critic.inputs) >= 3:
            state1_batch_with_action = state1_batch[:]
        else:
            state1_batch_with_action = [state1_batch]
        state1_batch_with_action.insert(agent.critic_action_input_idx, target_actions)
        target_q_values = agent.target_critic.predict_on_batch(state1_batch_with_action).flatten()
        assert target_q_values.shape == (agent.batch_size,)

        # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
        #  but only for the affected output units (as given by action_batch).
        discounted_reward_batch = agent.gamma * target_q_values
        discounted_reward_batch *= terminal1_batch
        assert discounted_reward_batch.shape == reward_batch.shape
        targets = (reward_batch + discounted_reward_batch).reshape(agent.batch_size, 1)

        # Perform a single batch update on the critic network.
        if len(agent.critic.inputs) >= 3:
            state0_batch_with_action = state0_batch[:]
        else:
            state0_batch_with_action = [state0_batch]
        state0_batch_with_action.insert(agent.critic_action_input_idx, action_batch)
        metrics = agent.critic.train_on_batch(state0_batch_with_action, targets)
        metricsss.append(metrics)

    # Update actor.
    #  TODO: implement metrics for actor
    if i > agent.nb_steps_warmup_actor:
        if len(agent.actor.inputs) >= 2:
            inputs = state0_batch[:]
        else:
            inputs = [state0_batch]
        if agent.uses_learning_phase:
            inputs += [agent.training]
        action_values = agent.actor_train_fn(inputs)[0]
        assert action_values.shape == (agent.batch_size, agent.nb_actions)

    if agent.target_model_update >= 1 and agent.step % agent.target_model_update == 0:
        agent.update_target_models_hard()




# test_step = 1000
# nihe_errors = []
# env.reset()
# # （forward）这个循环前向传播根据状态生成动作
# for i in range(test_step):
#     experience = agent.memory.sample(1)
#     observation = experience[0].state0[0]
#     state = agent.memory.get_recent_state(observation)
#     action = agent.select_action(state)  # TODO: move this into policy
#     # print("action", action)
#
#     # 测试actor训练拟合经验池情况
#     e_action = experience[0].action
#     # print("e_action", e_action)
#     nihe_error = np.square(action - e_action)
#     nihe_errors.append(nihe_error)
#
#     # 用gym环境测试actor训练效果
#     s_, r, d, _ = env.step(action)
#     env.render()


import matplotlib.pyplot as plt
metrics_names = agent.critic.metrics_names
plt.plot(np.array(metricsss)[:, 0], 'r', label=metrics_names[0])
plt.plot(np.array(metricsss)[:, 1], 'g', label=metrics_names[1])
plt.plot(np.array(metricsss)[:, 2], 'b', label=metrics_names[2])
# plt.plot(nihe_errors, 'y--', label="nihe_error")
plt.legend(loc="best")
plt.show()





# # agent.load_weights('ddpg_{}_weights.h5f'.format(ENV_NAME))
# agent.fit(env, nb_steps=10000, visualize=False, verbose=2)
#
# # After training is done, we save the final weights.
# # agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
#
# # Finally, evaluate our algorithm for 5 episodes.
# agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)

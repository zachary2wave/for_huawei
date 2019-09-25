import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym

EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 1
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 1
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization

class PPO(object):

    def __init__(self):
        # gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 50, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

        with tf.variable_scope('test'):
            self.tests = -tf.reduce_mean(tf.minimum(
                pi.prob(self.tfa) / oldpi.prob(self.tfa) * self.tfadv,
                tf.clip_by_value(pi.prob(self.tfa) / oldpi.prob(self.tfa), 1. - METHOD['epsilon'],
                                 1. + METHOD['epsilon']) * self.tfadv))

        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful
        # print(self.sess.run(self.tests, {self.tfs: s, self.tfa: a, self.tfadv: adv}))
        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            lossa = [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        lossc = [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]
        return  lossa, lossc

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 50, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def vars_generate1(self, scope_name_var):
        return [var for var in tf.global_variables() if scope_name_var in var.name]

    def get_weight(self):
        full_connect_variable = self.vars_generate1("pi")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())  ##一定要先初始化变量
            print(sess.run(full_connect_variable[0]))

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


dataPrimary = pd.read_csv("data/monitorChange.csv")
dataProgress = pd.read_csv("data/progress.csv")
ppo = PPO()
all_ep_r = []

buffer_s, buffer_a, buffer_r = [], [], []
ep_r = 0
env = gym.make('Pendulum-v0').unwrapped
s = env.reset()

for index, row in dataPrimary.iterrows():
    if index % 2000 == 0:
        if index > 100: all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
        print(
            'Ep: %i' % (index/2000),
            "|Ep_r: %i" % ep_r,
            ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
        )
        ep_r = 0
    buffer_s.append([row.obnext1, row.obnext2, row.obnext3])
    buffer_a.append(row.action)
    buffer_r.append((row.reward + 8) / 8)  # normalize reward, find to be useful
    ep_r += row.reward

    if index % 100 == 0:
        if index == 100: all_ep_r.append(ep_r)
        v_s_ = 0
        discounted_r = []
        for r in buffer_r[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()

        bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
        buffer_s, buffer_a, buffer_r = [], [], []
        lossa, lossc = ppo.update(bs, ba, br)
        print(lossa,lossc)
        # env.render()
        # # print(s)
        a = ppo.choose_action(s)
        # # print(a)
        # s_, r, done, _ = env.step(a)
        # s = s_
        # print(r)


plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode')
plt.ylabel('Moving averaged episode reward')
plt.show()


env = gym.make('Pendulum-v0').unwrapped
s = env.reset()
for i in range(10000):
    env.render()
    print(s)
    a = ppo.choose_action(s)
    print(a)
    s_, r, done, _ = env.step(a)
    s = s_
    print(r)
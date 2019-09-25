import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 1
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization

class PPO(object):

    def __init__(self):
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 50, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)
            params_c = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope("output_params_a"):
            self.params_a = pi_params
            with tf.variable_scope("output_params_c"):
                self.params_c = params_c
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
            temp=tf.constant(1.0)
#            self.opt = tf.train.AdamOptimizer(A_LR)
#            self.grads_vals = [(tf.clip_by_value(g,-1,1),v) for i, (g, v) in enumerate(self.opt.compute_gradients(self.aloss, self.params_a))]
#            self.atrain_op = self.opt.apply_gradients(self.grads_vals)
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful
        print("aloss:",self.sess.run(self.tests, {self.tfs: s, self.tfa: a, self.tfadv: adv}))
        # update actor
        #print("closs:",np.mean(adv*adv))
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
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]
            #print("梯度:", self.sess.run(self.grads_vals,{self.tfs: s, self.tfa: a, self.tfadv: adv}))
            #print("agent:",self.sess.run(self.params_a))

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r})for _ in range(C_UPDATE_STEPS)]
        #print("critic:",self.sess.run(self.params_c))


    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 50, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        #print(self.sess.run(self.params_a))
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

def random_batch(X_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train)-32, batch_size)
    X_batch = X_train.iloc[rnd_indices[0]:rnd_indices[0]+32]
    return X_batch

dataPrimary = pd.read_csv("../data/monitorChange.csv")
dataProgress = pd.read_csv("../data/progress.csv")
ppo = PPO()
all_ep_r = []
env = gym.make('Pendulum-v0').unwrapped
s1 = env.reset()
for ep in range(EP_MAX):
    data = random_batch(dataPrimary,batch_size=1)
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(32):  # in one episode
        env.render()
        a1 = ppo.choose_action(s1)
        s_1, r1, done, _ = env.step(a1)
        s1 = s_1
        print("测试环境:",r1)

        buffer_s.append(data[["obnext1","obnext2","obnext3"]].iloc[t].tolist())
        buffer_a.append(data["action"].iloc[t])
        buffer_r.append(data["reward"].iloc[t])  # normalize reward, find to be useful
        # update ppo
        if (t + 1) % BATCH == 0 or t == EP_LEN - 1:
            print(buffer_a)
            print(buffer_s)
            print(buffer_r)
            v_s_ = ppo.get_v(data[["obnext1","obnext2","obnext3"]].iloc[31])
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br)







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
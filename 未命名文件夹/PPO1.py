import os

import numpy as np
import pandas as pd
import gym

from baselines.ppo2.model import Model
from baselines.common.policies import build_policy

from baselines.common import tf_util as U



t_train_time = 10000
t_test_time = 10000

dataPrimary = pd.read_csv("data_p/monitorChange.csv")
dataProgress = pd.read_csv("data_p/progress.csv")

ob_space = (3,)
ac_space = (1,)
nb_action = 1

model_fn = Model
policy = build_policy(env, network, **network_kwargs)
model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                 nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                 max_grad_norm=max_grad_norm)
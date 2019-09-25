from RL_brain import DeepQNetwork
import pandas as pd
import numpy as np
import os
import gym
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train():

    step = 0

    for i in range(142877):
            RL.store_transition(dfa[i, :n_features], dfa[i, n_features], dfa[i, n_features+1], dfa[i, -n_features:])

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation

            # break while loop when end of this episode
            # if done:7
            #     break
            step += 1

    # end of game
    print('train over')


def run_():
    step = 0
    for episode in range(300):
        observation = env.reset()
        for t in range(1000):
            #env.render()
            #print(3)
            action = RL.choose_action(observation)
            #action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            #print(reward)
            observation = observation_
            step = step+1
            if done:
                break
    print("game over")
    print(step)
    #env.destroy()




if __name__ == "__main__":
    # maze game
    df = pd.read_csv("./cartpolechange.csv")
    dfa = np.array(df)
    n_actions = 2
    n_features = 4

    RL = DeepQNetwork(n_actions, n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    # env = gym.make('CartPole-v0')
    # run_()
    train()
    RL.plot_cost()
    output_q = RL.output(dfa)
    env = gym.make('CartPole-v0')
    run_()

    #print(output_q)


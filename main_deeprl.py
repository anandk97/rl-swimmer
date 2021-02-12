# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 15:14:39 2020

@author: Anand Krishnan
"""
import gym
import envs
from envs.utils import animate_gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import time


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.policy import BoltzmannGumbelQPolicy
from rl.memory import SequentialMemory
#%% 
env = gym.make('MicroSwimmer-v1')
# sim = gym.make('MicroSwimmer-v0')r
#%%
# env.reset()

# # env.render()

# total_rew = 0.0
# # total_rew = np.zeros((400,2))
# i = 0
# while True: 
    

#     # action_index = 0
#     state,rew,done,info = env.step(np.random.randint(0,8))
#     # print(state)
#     a = np.array([0, 0, state[6]])
#     b = np.array([state[2], state[3], 0])
#     cross_term = 0.5 * np.cross(a, b)
#     alignment = np.linalg.norm(cross_term[:2])
#     print(alignment)
#     dist = 100*(np.linalg.norm(env.x_naive[:2] - env.target) - np.linalg.norm(state[:2]- env.target))
#     print(dist)
#     total_rew += rew
#     # total_rew[i,:] = rew
#     if done:
#         break
#     i += 1
# print('The total reward is :',total_rew)
#%%

def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(14, activation='relu'))
    # model.add(Dense(14, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

states = 9

actions = 8 
model = build_model(states, actions)
model.summary()
#%%
def build_agent(model, actions):
   # policy = BoltzmannQPolicy()
    policy = BoltzmannGumbelQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, target_model_update=1e-3)
    return dqn
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
# dqn.fit(env, nb_steps=1e6,visualize=False, verbose=1)
# dqn.save_weights('dqn_weights_Dec7.h5f', overwrite=True)
#%%
dqn.load_weights('dqn_weights_final.h5f')
scores = dqn.test(env, nb_episodes=100, visualize=False)

# print(np.mean(scores.history['episode_reward']))

#%% Plotting
fig = plt.figure()

ax = plt.axes(xlim=(0, 1), ylim=(0, 1), xlabel='x', ylabel='y')
clev = 15
cont = plt.contourf(env.xx, env.yy, env.omg_precompute[:, :, 0], clev)  # first image on screen
animate_lam = lambda j: animate_gym(j, env.xx, env.yy, env.omg_precompute, env.rl_traj, env.rl_ctrl, env.naive_traj, env.naive_ctrl, env.target)
anim = anm.FuncAnimation(fig, animate_lam, frames=env.t, interval=1, repeat=False)
plt.tight_layout()
plt.show()
# Uncomment this to save the results a gif
# f = r"C:\Users\Anand Krishnan\Desktop\RL\Work in progress\Gym environment\Deep RL\success6.gif"
# writergif = anm.PillowWriter(fps=30) 
# anim.save(f, writer=writergif)

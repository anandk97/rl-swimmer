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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.policy import MaxBoltzmannQPolicy
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
#%% 
env = gym.make('MicroSwimmer-v1')
# sim = gym.make('MicroSwimmer-v0')
#%%

def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(14, activation='relu'))
    # model.add(Dense(14, activation='relu'))
    model.add(Dense(actions, activation='relu'))
    return model

states = 9

actions = 8 
model = build_model(states, actions)
model.summary()
#%%
def build_agent(model, actions):
    # policy = BoltzmannQPolicy()
    # policy = MaxBoltzmannQPolicy()
    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, target_model_update=1e-3)
    return dqn
dqn = build_agent(model, actions)
opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
dqn.compile(opt, metrics=['mse'])
# dqn.load_weights('dqn_weights_May19_flow2.h5f')
#%%
history = dqn.fit(env, nb_steps=5e3,visualize=False, verbose=1)
#%%
# dqn.save_weights('dqn_train_5000.h5f', overwrite=True)
#%%
dqn.load_weights('dqn_weights_May11_flow1.h5f')
# dqn.load_weights('dqn_train_2500.h5f')
scores = dqn.test(env, nb_episodes=1000, visualize=False)   
#%%
higher = 0
equal = 0
lower = 0
for episode_reward in scores.history['episode_reward']:
    if episode_reward > 0.2:
        higher += 1
    elif -0.2 < episode_reward < 0.2:
        equal += 1
    else: 
        lower += 1

print(higher)
print(equal)
print(lower)
# print(np.mean(scores.history['episode_reward']))

#%% Plotting
fig = plt.figure()

ax = plt.axes(xlim=(0, 1), ylim=(0, 1), xlabel='x', ylabel='y')
clev = 15
cont = plt.contourf(env.xx, env.yy, env.omg_precompute[:, :, 0], clev,cmp = plt.cm.gray)  # first image on screen
animate_lam = lambda j: animate_gym(j, env.xx, env.yy, env.omg_precompute, env.rl_traj, env.rl_ctrl, env.naive_traj, env.naive_ctrl, env.target)
anim = anm.FuncAnimation(fig, animate_lam, frames=env.t, interval=1, repeat=False)
plt.tight_layout()
plt.show()
# Uncomment this to save the results a gif
# f = r"C:\Users\Anand Krishnan\Desktop\RL\Work in progress\Gym environment\Deep RL\success6.gif"
# writergif = anm.PillowWriter(fps=30) 
# anim.save(f, writer=writergif)
#%%
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
#%% Plotting images
fig = plt.figure()

ax = plt.axes(xlim=(0, 1), ylim=(0, 1), xlabel='x', ylabel='y')
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
clev = 15
cont = plt.contourf(env.xx, env.yy, env.omg_precompute[:, :, 0], clev,cmap = plt.cm.gray)  # first image on screen
plt.colorbar()
# # Position of the target
plt.scatter(env.target[0], env.target[1], color='red')
# Current position of the RL microswimmer
for j in range(1,3):#range(len(env.rl_traj)):
    plt.scatter(env.rl_traj[j][0],env.rl_traj[j][1], color=lighten_color('blue',0.5),s=0.5)
    plt.scatter(env.naive_traj[j][0], env.naive_traj[j][1], color=lighten_color('green',1),s=15)    

# plt.scatter(env.naive_traj[0][0], env.naive_traj[0][1], color='#1f77b4')    
#%% Contour plot imaging
fig = plt.figure()

# ax = plt.axes(xlim=(0.25, 0.75), ylim=(0.25, 0.75), xlabel='x', ylabel='y')
# ax = plt.axes(xlim=(0, 1), ylim=(0, 1), xlabel='x', ylabel='y')
ax = plt.axes(xlim=(0.4, 1), ylim=(0.4, 1), xlabel='x', ylabel='y')
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 15

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
# plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
clev = 15
cont = plt.contourf(env.xx, env.yy, env.omg_precompute[:, :, 0], clev,cmap = plt.cm.gray)  # first image on screen
plt.colorbar()
plt.scatter(env.target[0], env.target[1], color='red')
state = np.zeros(9)
# for i in range(7):
#     for j in range(7):
#         x = (i/14) + 0.3
#         y = (j/14) + 0.3
for i in range(10):
    for j in range(10):
        x = (i/10)+0.05
        y = (j/10)+0.05
        cur_pos = np.array([x,y])
        target = np.array([0.5,0.5])
        direction = env.target_direction(cur_pos,target)
        target_direction = direction.copy()
        target_distance = np.linalg.norm(cur_pos - target)
        vorticity = env.bilinear(cur_pos[0],cur_pos[1],env.xx,env.yy,env.omg)
        t = 0
        state[:2] = cur_pos
        state[2:4] = direction
        state[4:6] = target_direction
        state[6] = target_distance
        state[7] = vorticity
        state[8] = t
        action = dqn.forward(state)
        u = target_direction[0]*np.cos(action*(np.pi/4))
        v = target_direction[1]*np.sin(action*(np.pi/4))
        plt.quiver(x,y,u,v)



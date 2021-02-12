# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 15:50:17 2020

@author: Anand Krishnan
"""
import gym
import envs
from envs.utils import animate_gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import time
#%%
env = gym.make('MicroSwimmer-v0')
# sim = gym.make('MicroSwimmer-v0')
#%%
# env.reset()
# adv = [env.state.x_adv[:1]]
# adv1 = [env.state.x_adv[1:2]]
# rl = [env.state.x_rl[:1]]
# rl1 = [env.state.x_rl[1:2]]
# while True:
#     state,rew, done = env.step(0)
#     if done:
#         break
#     adv.append(env.state.x_adv[:1])
#     adv1.append(env.state.x_adv[1:2])
#     rl.append(env.state.x_rl[:1])
#     rl1.append(env.state.x_rl[1:2])
# plt.plot(adv,adv1,color ='blue')
# # plt.plot(rl,rl1,color = 'red')
#%%
env.reset()

env.render()
Q = np.load('Q3_Dec5_5000_eps.npy')
total_rew = 0.
theta_sts = [env.state.theta_st]
dist_sts = [env.state.dist_st]
idx = [1]
while True: 
    
    action_index = np.argmax(Q[env.state.dist_st,env.state.omg_st,env.state.theta_st,:])
    # action_index = 0
    state,rew,done = env.step(action_index)
    theta_sts.append(state.theta_st)
    dist_sts.append(state.dist_st)
    idx.append(idx[-1]+1)
    total_rew += rew

    if done:
        break

print('The total reward is :',total_rew)

#%%
fig = plt.figure()

ax = plt.axes(xlim=(0, 1), ylim=(0, 1), xlabel='x', ylabel='y')
clev = 15
cont = plt.contourf(env.xx, env.yy, env.omg_precompute[:, :, 0], clev)  # first image on screen
animate_lam = lambda j: animate_gym(j, env.xx, env.yy, env.omg_precompute, env.rl_traj, env.rl_ctrl, env.naive_traj, env.naive_ctrl, env.target)
anim = anm.FuncAnimation(fig, animate_lam, frames=env.state.t, interval=1, repeat=False)
plt.tight_layout()
plt.show()
# Uncomment this to save the results a gif
# =============================================================================
# f = r"C:\Users\Anand Krishnan\Desktop\RL\Results\Gifs\Static_flow.gif"
# writergif = anm.PillowWriter(fps=30) 
# anim.save(f, writer=writergif)
# =============================================================================
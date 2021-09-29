# utils.py
# Modified August 4th 2020
import numpy as np
import math
import scipy as sp
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp2d
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

def bilinear(xq, yq, xx, yy, f):
    x = np.array([xx[0, :]])
    y = np.array([yy[:, 0]])
    interp_spline = RectBivariateSpline(y, x, f, kx=1, ky=1)
    fq = interp_spline(yq, xq)
    return fq


def target_direction(current_pos, target):
    o_hat = np.zeros((2, 1))
    o_hat = target - current_pos
    o_hat = o_hat / (np.linalg.norm(o_hat)+1e-8)
    return o_hat

def euler_update_position(cur_pos, cur_direc, o_hat, omg, dt, V_s_tilde, B, xx, yy, u, v):
    next_pos = np.zeros(4)
    total_vel = np.sqrt(u ** 2 + v ** 2)
    U_rms = np.sqrt(np.mean(total_vel ** 2))
    V_s = V_s_tilde * U_rms
    next_pos[:2] = cur_pos + dt*dpos_dt(cur_pos,xx,yy,u,v,cur_direc,V_s)
    next_direc = cur_direc + dt*ddirec_dt(cur_pos,cur_direc,omg,B,o_hat,xx,yy)
    next_pos[2:] = next_direc/np.linalg.norm(next_direc)
    return next_pos

def update_position(cur_pos, cur_direc, o_hat, omg, dt, V_s_tilde, B, xx, yy, u, v):
    # Updates position and direction based on the differential equations
    # Using Runge-Kutta 4th order method
    next_pos = np.zeros(4)
    total_vel = np.sqrt(u ** 2 + v ** 2)
    U_rms = np.sqrt(np.mean(total_vel ** 2))
    V_s = V_s_tilde * U_rms
    # Position update
    k1 = dt * dpos_dt(cur_pos, xx, yy, u, v, cur_direc, V_s)
    k2 = dt * dpos_dt(cur_pos + (k1 / 2), xx, yy, u, v, cur_direc, V_s)
    k3 = dt * dpos_dt(cur_pos + (k2 / 2), xx, yy, u, v, cur_direc, V_s)
    k4 = dt * dpos_dt(cur_pos + k3, xx, yy, u, v, cur_direc, V_s)
    next_pos[:2] = cur_pos + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 4
    # Direction update
    k1 = dt * ddirec_dt(cur_pos, cur_direc, omg, B, o_hat,xx,yy)
    k2 = dt * ddirec_dt(cur_pos, cur_direc + (k1 / 2), omg, B, o_hat,xx,yy)
    k3 = dt * ddirec_dt(cur_pos, cur_direc + (k2 / 2), omg, B, o_hat,xx,yy)
    k4 = dt * ddirec_dt(cur_pos, cur_direc + k3, omg, B, o_hat,xx,yy)
    next_direc = cur_direc + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 4
    next_pos[2:] = next_direc/np.linalg.norm(next_direc)
    return next_pos


def ddirec_dt(cur_pos, cur_direc, omg, B, o_hat,xx,yy):
    # Returns rate of change of direction
    omg_xy = bilinear(cur_pos[0], cur_pos[1], xx, yy, omg)
    omg_xy = np.float(omg_xy)
    a = np.array([0, 0, omg_xy])
    b = np.array([cur_direc[0], cur_direc[1], 0])
    cross_term = 0.5 * np.cross(a, b)
    d_direc = np.zeros(2)
    c = np.dot(o_hat, cur_direc)
    # c = c/np.linalg.norm(c)
    d_direc = (1 / (2 * B)) * (o_hat - c * cur_direc) + cross_term[:2]
    return d_direc


def dpos_dt(cur_pos, xx, yy, u, v, cur_direc, V_s):
    # Returns rate of change of position
    Ux = bilinear(cur_pos[0], cur_pos[1], xx, yy, u)
    Ux = np.float(Ux)
    Uy = bilinear(cur_pos[0], cur_pos[1], xx, yy, v)
    Uy = np.float(Uy)
    u_xy = np.array([Ux, Uy])
    rate = u_xy + V_s * cur_direc
    return rate

def animate_gym(i, xx, yy, omg_precompute, x_rl, udir_rl, x_naive, udir_naive, target):
    # Plots the changing position and direction of the RL and naive microswimmer
    # Fluid flow plot
    clev = 15
    plt.clf()
    cont = plt.contourf(xx, yy, omg_precompute[:, :, i], clev,cmap=plt.cm.gray)
    #cont = plt.contourf(xx, yy, omg_precompute[:, :, 0], clev)
    plt.colorbar()
    # Position of the target
    plt.scatter(target[0], target[1], color='red')
    # Current position of the RL microswimmer
    plt.scatter(x_rl[i][0], x_rl[i][1], color='blue')
    # Current direction of the RL microswimmer
    # plt.quiver(x_rl[i][0], x_rl[i][1], x_rl[i][2], x_rl[i][3], color='blue')
    plt.quiver(x_rl[i][0], x_rl[i][1], 0.5,0.5, color='blue')
    # Current position of the naive microswimmer
    plt.scatter(x_naive[i][0], x_naive[i][1], color='green')
    # Current direction of the microswimmer
    # plt.quiver(x_naive[i][0], x_naive[i][1], x_naive[i][2], x_naive[i][3], color='green')
    plt.quiver(x_naive[i][0], x_naive[i][1], 0.5,0.5, color='green')
    # RL Control direction
    plt.quiver(x_rl[i][0], x_rl[i][1], udir_rl[i][0], udir_rl[i][1], color='black')
    # Naive Control direction
    plt.quiver(x_naive[i][0], x_naive[i][1], udir_naive[i][0], udir_naive[i][1], color='black')
    return cont

def animate_rl(i, xx, yy, omg_precompute, x_rl, udir_rl, x_naive, udir_naive, target):
    # Plots the changing position and direction of the RL and naive microswimmer
    # Fluid flow plot
    clev = 15
    plt.clf()
    cont = plt.contourf(xx, yy, omg_precompute[:, :, i], clev)
    #cont = plt.contourf(xx, yy, omg_precompute[:, :, 0], clev)
    plt.colorbar()
    # Position of the target
    plt.scatter(target[0], target[1], color='red')
    # Current position of the RL microswimmer
    plt.scatter(x_rl[0, i], x_rl[1, i], color='blue')
    # Current direction of the RL microswimmer
    plt.quiver(x_rl[0, i], x_rl[1, i], x_rl[2, i], x_rl[3, i], color='blue')
    # Current position of the naive microswimmer
    plt.scatter(x_naive[0, i], x_naive[1, i], color='green')
    # Current direction of the microswimmer
    plt.quiver(x_naive[0, i], x_naive[1, i], x_naive[2, i], x_naive[3, i], color='green')
    # RL Control direction
    plt.quiver(x_rl[0, i], x_rl[1, i], udir_rl[0, i], udir_rl[1, i], color='black')
    # Naive Control direction
    plt.quiver(x_naive[0, i], x_naive[1, i], udir_naive[0, i], udir_naive[1, i], color='black')
    return cont
def get_action_list(o_hat):
    rot_90 = np.array([[0, 1], [-1, 0]])
    o_hat_perp = np.dot(rot_90, o_hat)
    actions = np.stack((o_hat, -o_hat, o_hat_perp, -o_hat_perp), axis=1)
    return actions


def theta_state(o_hat, cur_direc):
    # Inputs : o_hat = Normalized Vector pointing to target
    #          cur_direc = Normalized vector of current_direction
    T_3d = np.append(o_hat, [0])
    p_3d = np.append(cur_direc, [0])
    c = np.cross(T_3d, p_3d)
    if c[-1] < 0:
        theta_xy = -math.atan2(np.linalg.norm(c), np.dot(o_hat, cur_direc))
    else:
        theta_xy = math.atan2(np.linalg.norm(c), np.dot(o_hat, cur_direc))
    angle_45 = math.pi / 4
    if -angle_45 <= theta_xy < angle_45:
        theta_st = 0  # red state
    elif angle_45 <= theta_xy < 3 * angle_45:
        theta_st = 1  # orange state
    elif -3 * angle_45 <= theta_xy < -angle_45:
        theta_st = 2  # blue state
    else:
        theta_st = 3  # gray state
    return theta_st


def omega_state(cur_pos, xx, yy, omg, omg_0):
    omg_xy = bilinear(cur_pos[0], cur_pos[1], xx, yy, omg)
    if omg_xy > omg_0:
        omg_st = 0
    elif -omg_0 <= omg_xy <= omg_0:
        omg_st = 1
    else:
        omg_st = 2
    return omg_st

def ppms_policy(omg_st,theta_st):
    if theta_st == 3:
        if omg_st == 2:
            action = 1
        elif omg_st == 1:
            action = 0
        else:
            action = 2
    elif theta_st == 2:
        if omg_st == 2:
            action = 2
        elif omg_st == 1:
            action = 3
        else:
            action = 0
    elif theta_st == 1:
        if omg_st == 2:
            action = 0
        elif omg_st == 1:
            action = 0
        else:
            action = 1
    elif theta_st == 0:
        if omg_st == 2:
            action = 0
        elif omg_st == 1:
            action = 1
        else:
            action = 1 
    return action
def get_random_position(lower_bound=10, upper_bound=90):
    p = np.random.randint(lower_bound, upper_bound)
    q = np.random.randint(lower_bound, upper_bound)
    pos = np.array([p / 100, q / 100])
    return pos

def get_pseudo_random_position(lower_bound = 10,upper_bound = 90,seed = 5):
    random.seed(seed)
    p = random.randint(lower_bound, upper_bound)
    q = random.randint(lower_bound, upper_bound)
    pos = np.array([p / 100, q / 100])
    return pos

def get_random_direction(lower_bound=10, upper_bound=90):
    p = np.random.randint(lower_bound, upper_bound)
    q = np.random.randint(lower_bound, upper_bound)
    direction = np.array([p / 100, q / 100])
    direction /= np.linalg.norm(direction)
    return direction

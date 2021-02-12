import gym
from gym import spaces
from gym import logger
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import time
import math
import scipy as sp
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp2d

class FlowState:
    """
    the tetris state
    """
    def __init__(self,u, v, omg, x_rl, x_adv, x_naive,udir_rl,udir_naive,t,omg_st,theta_st,dist_st):

        # The x and y velocity components
        self.u = u
        self.v = v
        # The vorticity
        self.omg = omg      
        self.x_rl = x_rl
        self.x_adv = x_adv
        self.x_naive = x_naive
        self.t = t
        self.udir_rl = udir_rl
        self.udir_naive = udir_naive
        self.omg_st = omg_st
        self.theta_st = theta_st
        self.dist_st = dist_st
        
    def copy(self):
        return FlowState(
            self.u.copy(),
            self.v.copy(),
            self.omg.copy(),
            self.x_rl.copy(),
            self.x_adv.copy(),
            self.x_naive.copy(),
            self.udir_rl.copy(),
            self.udir_naive.copy(),
            self.t,
            self.omg_st,
            self.theta_st,
            self.dist_st
        )


class MicroSwimmerEnv(gym.Env):

    def __init__(self):
        data = loadmat('precompute4000_equal_vortices.mat')
        # precompute1000_diagonal_c_vortices.mat
        # Extract data from precomputed flow file
        self.xx = np.array(data['xx'])
        self.yy = np.array(data['yy'])
        self.u_precompute = np.array(data['u_precompute'])
        self.v_precompute = np.array(data['v_precompute'])
        self.omg_precompute = np.array(data['omg_precompute'])
        omg_rms = np.float(np.array(data['omg_rms']))
        nx = 128
        ny = nx
        Lx = 1
        Ly = 1
        dx = Lx / nx
        dy = Ly / ny
        self.V_s_tilde = 2.0
        B_tilde = 0.01
        self.B = B_tilde / omg_rms
        self.dt = 1 / nx / 4
        self.state_space = spaces.Box(np.array([0,0]), np.array([1,1]))
        self.target = np.array([0,0])
        self.state = None
        self.rl_traj = []
        self.naive_traj = []
        self.naive_ctrl = []
        self.rl_ctrl = []
        
    def step(self, action_idx):
        
        self.state.t += 1
        rl_cur_pos = self.state.x_rl[0:2]
        rl_cur_direc = self.state.x_rl[2:]

        # adv_cur_pos = self.state.x_adv[0:2]
        # adv_cur_direc = self.state.x_adv[2:]
        
        naive_cur_pos = self.state.x_naive[0:2]
        naive_cur_direc = self.state.x_naive[2:]
        
        if np.linalg.norm(rl_cur_pos - self.target) < 0.03 or np.linalg.norm(naive_cur_pos - self.target) < 0.03:
            return self.state, self._get_reward(), True
            
        if (not self.state_space.contains(self.state.x_rl[:2]) or not self.state_space.contains(self.state.x_naive[:2])):
            return self.state, self._get_reward(), True
        
        # Get Naive controller input

        naive_ctrl_direc = self.target_direction(naive_cur_pos, self.target)
        self.state.udir_naive = naive_ctrl_direc
        naive_next_state = self.euler_update_position(naive_cur_pos, naive_cur_direc, naive_ctrl_direc, self.state.omg, self.dt, self.V_s_tilde,
                                           self.B, self.xx, self.yy, self.state.u, self.state.v)
        self.state.x_naive = naive_next_state
        self.naive_traj.append(self.state.x_naive)
        self.naive_ctrl.append(self.state.udir_naive)
        # # Get adversarial controller input
        
        # adv_ctrl_direc = self.target_direction(adv_cur_pos, self.target)
        # adv_next_state = self.euler_update_position(adv_cur_pos, adv_cur_direc, adv_ctrl_direc, self.state.omg, self.dt, self.V_s_tilde,
        #                                     self.B, self.xx, self.yy, self.state.u, self.state.v)
        # self.state.x_adv = adv_next_state
        # Get Reinforcement Learning controller input
        
        rl_target_direc = self.target_direction(rl_cur_pos, self.target)
        actions = self.get_actions(rl_target_direc)
        
        rl_ctrl_direc = actions[:, action_idx]
        self.state.udir_rl = rl_ctrl_direc
        rl_next_state = self.euler_update_position(rl_cur_pos, rl_cur_direc, rl_ctrl_direc, self.state.omg, self.dt, self.V_s_tilde,
                                        self.B, self.xx, self.yy, self.state.u, self.state.v)
        self.state.x_rl = rl_next_state
        self.rl_traj.append(self.state.x_rl)
        self.rl_ctrl.append(self.state.udir_rl)
        
        omg_st = self.omega_state(self.state.x_rl[:2],self.xx,self.yy, self.state.omg)
        # theta_st = self.theta_state(self.target_direction(self.state.x_rl[:2], self.target),self.state.x_rl[2:])
        theta_st = self.theta_state(self.state.x_rl[:2])
        dist_st = self.distance_state(self.state.x_rl[:2])
        
        reward = self._get_reward()
        
        # if omg_st != self.state.omg_st or theta_st != self.state.theta_st:
        #     self.state.x_adv[:2] = rl_next_state[:2]
            
        self.state.omg_st = omg_st
        self.state.theta_st = theta_st
        self.state.dist_st = dist_st
        
        self.state.u = self.u_precompute[:,:,self.state.t]
        self.state.v = self.v_precompute[:,:,self.state.t]
        self.state.omg = self.omg_precompute[:,:,self.state.t]
        

        return self.state.copy(), reward, False

    def reset(self):
        u = self.u_precompute[:,:,0]
        v = self.v_precompute[:,:,0]
        omg = self.omg_precompute[:,:,0]
        
        x_rl = np.zeros(4)
        x_naive = np.zeros(4)
        
        # self.target = self.get_random_position()
        self.target = np.array([0.5,0.5])
        
        # x_rl[:2] = self.get_random_position()
        radius = 0.45
        x_rl[:2] = self.get_circular_position(radius)
        # x_rl[:2] = np.array([0.9,0.1])
        x_rl[2:] = self.target_direction(x_rl[:2], self.target)
        x_adv = x_rl.copy()
        x_naive = x_rl.copy()
        udir_rl = x_rl[2:]
        udir_naive = udir_rl.copy()
        t = 0
        omg_st = self.omega_state(x_rl[:2],self.xx,self.yy, omg)
        # theta_st =self.theta_state(x_rl[2:],x_rl[2:])
        theta_st = self.theta_state(x_rl[:2])
        dist_st = self.distance_state(x_rl[:2])
        
        self.state = FlowState(u, v, omg, x_rl, x_adv, x_naive,udir_rl,udir_naive,t,omg_st,theta_st,dist_st)
        self.rl_traj = []
        self.naive_traj = []
        self.naive_ctrl = []
        self.rl_ctrl = []
        self.naive_traj.append(self.state.x_naive)
        self.naive_ctrl.append(self.state.udir_naive)
        self.rl_traj.append(self.state.x_rl)
        self.rl_ctrl.append(self.state.udir_rl)
        return self.state.copy()

    def render(self):
        clev = 15
        plt.clf()
        cont = plt.contourf(self.xx, self.yy, self.omg_precompute[:, :, self.state.t], clev)
        plt.colorbar()
        # Position of the target
        plt.scatter(self.target[0], self.target[1], color='red')
        # Current position of the RL microswimmer
        plt.scatter(self.state.x_rl[0], self.state.x_rl[1], color='blue')
        # Current direction of the RL microswimmer
        plt.quiver(self.state.x_rl[0], self.state.x_rl[1], self.state.x_rl[2], self.state.x_rl[3], color='blue')
        # Current position of the naive microswimmer
        plt.scatter(self.state.x_naive[0], self.state.x_naive[1], color='green')
        # Current direction of the microswimmer
        plt.quiver(self.state.x_naive[0], self.state.x_naive[1], self.state.x_naive[2], self.state.x_naive[3], color='green')
        # RL Control direction
        plt.quiver(self.state.x_rl[0], self.state.x_rl[1], self.state.udir_rl[0], self.state.udir_rl[1], color='black')
        # Naive Control direction
        plt.quiver(self.state.x_naive[0], self.state.x_naive[1], self.state.udir_naive[0], self.state.udir_naive[1], color='black')
    
    def animate(self):
        fig = plt.figure()
        ax = plt.axes(xlim=(0, 1), ylim=(0, 1), xlabel='x', ylabel='y')
        clev = 15
        cont = plt.contourf(self.xx, self.yy, self.omg_precompute[:, :, 0], clev)  # first image on screen
        animate_lam = lambda j: animate_rl(j, self.xx, self.yy, self.omg_precompute, self.state.x_rl, self.state.udir_rl, self.state.x_naive, self.state.udir_naive, self.target)
        anim = anm.FuncAnimation(fig, animate_lam, frames=self.state.t, interval=1, repeat=False)
        plt.tight_layout()
        plt.show()
        # Uncomment this to save the results a gif
        # =============================================================================
        # f = r"C:\Users\Anand Krishnan\Desktop\RL\Results\Gifs\Static_flow.gif"
        # writergif = anm.PillowWriter(fps=30) 
        # anim.save(f, writer=writergif)
        # =============================================================================
    def _get_reward(self):
        """
        reward function
        """
        # rew = 128*(np.linalg.norm(self.state.x_adv[:2] - self.target) - np.linalg.norm(self.state.x_rl[:2]-self.target))
        rew = 128*(np.linalg.norm(self.state.x_naive[:2] - self.target) - np.linalg.norm(self.state.x_rl[:2]-self.target))
        return rew

    def get_actions(self,target_direc):
        rot_45 = np.array([[np.cos(np.pi/4),np.sin(np.pi/4)],[-np.sin(np.pi/4),np.cos(np.pi/4)]])
        rot_90 = np.array([[0, 1], [-1, 0]])
        o_45 = np.dot(rot_45, target_direc)
        o_135 = np.dot(rot_90, o_45)
        o_90 = np.dot(rot_90, target_direc)
        actions = np.stack((target_direc, o_45, o_90,o_135,-target_direc,-o_45,-o_90,-o_135), axis=1)
        return actions

    def bilinear(self,xq, yq, xx, yy, f):
        x = np.array([xx[0, :]])
        y = np.array([yy[:, 0]])
        interp_spline = RectBivariateSpline(y, x, f, kx=1, ky=1)
        fq = interp_spline(yq, xq)
        return fq
    
    def target_direction(self,current_pos, target):
        target_direc = np.zeros((2, 1))
        target_direc = target - current_pos
        target_direc = target_direc / (np.linalg.norm(target_direc)+1e-8)
        return target_direc
    
    def euler_update_position(self,cur_pos, cur_direc, target_direc, omg, dt, V_s_tilde, B, xx, yy, u, v):
        next_pos = np.zeros(4)
        total_vel = np.sqrt(u ** 2 + v ** 2)
        U_rms = np.sqrt(np.mean(total_vel ** 2))
        V_s = V_s_tilde * U_rms
        next_pos[:2] = cur_pos + dt*self.dpos_dt(cur_pos,xx,yy,u,v,cur_direc,V_s)
        next_direc = cur_direc + dt*self.ddirec_dt(cur_pos,cur_direc,omg,B,target_direc,xx,yy)
        next_pos[2:] = next_direc/np.linalg.norm(next_direc)
        return next_pos
    
    def ddirec_dt(self,cur_pos, cur_direc, omg, B, target_direc,xx,yy):
        # Returns rate of change of direction
        omg_xy = self.bilinear(cur_pos[0], cur_pos[1], xx, yy, omg)
        omg_xy = np.float(omg_xy)
        a = np.array([0, 0, omg_xy])
        b = np.array([cur_direc[0], cur_direc[1], 0])
        cross_term = 0.5 * np.cross(a, b)
        d_direc = np.zeros(2)
        c = np.dot(target_direc, cur_direc)
        # c = c/np.linalg.norm(c)
        d_direc = (1 / (2 * B)) * (target_direc - c * cur_direc) + cross_term[:2]
        return d_direc
    
    def dpos_dt(self,cur_pos, xx, yy, u, v, cur_direc, V_s):
        # Returns rate of change of position
        Ux = self.bilinear(cur_pos[0], cur_pos[1], xx, yy, u)
        Ux = np.float(Ux)
        Uy = self.bilinear(cur_pos[0], cur_pos[1], xx, yy, v)
        Uy = np.float(Uy)
        u_xy = np.array([Ux, Uy])
        rate = u_xy + V_s * cur_direc
        return rate
    
    def get_random_position(self,lower_bound=10, upper_bound=90):
        p = np.random.randint(lower_bound, upper_bound)
        q = np.random.randint(lower_bound, upper_bound)
        pos = np.array([p / 100, q / 100])
        return pos
    
    def get_circular_position(self,radius):
        theta = np.random.randint(0,359)
        pos = np.zeros(2)
        pos[0] = self.target[0] + radius*np.cos((theta/180)*np.pi)
        pos[1] = self.target[1] + radius*np.sin((theta/180)*np.pi)
        return pos
    
    def set_state_and_target(self, state,target):
        """
        set the field and the next piece
        """
        self.target = target
        self.state = state.copy()
    
    def omega_state(self,cur_pos, xx, yy, omg):
        omg_xy = self.bilinear(cur_pos[0], cur_pos[1], xx, yy, omg)

        if omg_xy > 30:
            omg_st = 0
        elif 10 < omg_xy <= 30:
            omg_st = 1
        elif 0 < omg_xy <= 10:
            omg_st = 2
        elif -10 < omg_xy <= 0:
            omg_st = 3
        elif -30 < omg_xy <= -10:
            omg_st = 4
        else:
            omg_st = 5
        return omg_st
    
    def distance_state(self,cur_pos):
    
        if np.linalg.norm(cur_pos - self.target) < 0.03:
            dist_st = 0
        elif 0.03 <= np.linalg.norm(cur_pos - self.target) < 0.1:
            dist_st = 1
        elif 0.1 <= np.linalg.norm(cur_pos - self.target) < 0.3:
            dist_st = 2
        elif 0.3 <= np.linalg.norm(cur_pos - self.target) < 0.5:
            dist_st = 3
        elif 0.5 <= np.linalg.norm(cur_pos - self.target) < 0.7:
            dist_st = 4
        elif 0.7 <= np.linalg.norm(cur_pos - self.target) < 0.9:
            dist_st = 5
        else:
            dist_st = 6
            
        return dist_st
    # def theta_state(self,target_direc, cur_direc):
    def theta_state(self, cur_pos):
        # Inputs : target_direc = Normalized Vector pointing to target
        #          cur_direc = Normalized vector of current_direction
        # T_3d = np.append(target_direc, [0])
        # p_3d = np.append(cur_direc, [0])
        # c = np.cross(T_3d, p_3d)
        # if c[-1] < 0:
        #     theta_xy = -math.atan2(np.linalg.norm(c), np.dot(target_direc, cur_direc))
        # else:
        #     theta_xy = math.atan2(np.linalg.norm(c), np.dot(target_direc, cur_direc))
        x = self.target[0] - cur_pos[0]
        y = self.target[1] - cur_pos[1]
        theta_xy = np.arctan2(y,x)
        angle_45 = math.pi / 4
        if 0 < theta_xy <= angle_45:
            theta_st = 0  
        elif angle_45 < theta_xy <= 2 * angle_45:
            theta_st = 1  
        elif 2 * angle_45 < theta_xy <= 3 * angle_45:
            theta_st = 2
        elif 3 * angle_45 < theta_xy <= 4 * angle_45:
            theta_st = 3
        elif -angle_45 < theta_xy <= 0:
            theta_st = 4
        elif -2*angle_45 < theta_xy <= -angle_45:
            theta_st = 5
        elif -3 * angle_45 < theta_xy <= - 2 * angle_45:
            theta_st = 6  
        else:
            theta_st = 7  
        return theta_st



class DeepMicroSwimmerEnv(gym.Env):

    def __init__(self):
        data = loadmat('precompute4000_equal_vortices.mat')
        # precompute1000_diagonal_c_vortices.mat
        # Extract data from precomputed flow file
        self.xx = np.array(data['xx'])
        self.yy = np.array(data['yy'])
        self.u_precompute = np.array(data['u_precompute'])
        self.v_precompute = np.array(data['v_precompute'])
        self.omg_precompute = np.array(data['omg_precompute'])
        omg_rms = np.float(np.array(data['omg_rms']))
        nx = 128
        ny = nx
        Lx = 1
        Ly = 1
        dx = Lx / nx
        dy = Ly / ny
        self.V_s_tilde = 1.5
        B_tilde = 0.01
        self.B = B_tilde / omg_rms
        self.dt = 1 / nx / 4
        self.state_space = spaces.Box(np.array([0,0]), np.array([1,1]))
        self.target = np.array([0,0])
        self.state = None
        self.rl_traj = []
        self.naive_traj = []
        self.naive_ctrl = []
        self.rl_ctrl = []
        # The x and y velocity components
        self.u = None
        self.v = None
        # The vorticity
        self.omg = None
        self.t = None
        self.x_naive = None
        
        self.udir_rl = None
        self.udir_naive = None
    def step(self, action_idx):
        
        self.t += 1
        rl_cur_pos = self.state[0:2]
        rl_cur_direc = self.state[2:4]

        
        naive_cur_pos = self.x_naive[0:2]
        naive_cur_direc = self.x_naive[2:]
        
        if np.linalg.norm(rl_cur_pos - self.target) < 0.03 or np.linalg.norm(naive_cur_pos - self.target) < 0.03:
            # return self.state, self.reward + 1e2, True, {}
            return self.state, self._get_reward(), True, {}    
        
        if (not self.state_space.contains(self.state[:2]) or not self.state_space.contains(self.x_naive[:2])):
            # return self.state, self.reward -1e7, True, {}
            return self.state, self._get_reward(), True, {}
        # Get Naive controller input

        naive_ctrl_direc = self.target_direction(naive_cur_pos, self.target)
        self.udir_naive = naive_ctrl_direc
        naive_next_state = self.euler_update_position(naive_cur_pos, naive_cur_direc, naive_ctrl_direc, self.omg, self.dt, self.V_s_tilde,
                                           self.B, self.xx, self.yy, self.u, self.v)
        self.x_naive = naive_next_state
        self.naive_traj.append(self.x_naive)
        self.naive_ctrl.append(self.udir_naive)

        
        rl_target_direc = self.target_direction(rl_cur_pos, self.target)
        actions = self.get_actions(rl_target_direc)
        
        rl_ctrl_direc = actions[:, action_idx]
        self.udir_rl = rl_ctrl_direc
        rl_next_state = self.euler_update_position(rl_cur_pos, rl_cur_direc, rl_ctrl_direc, self.omg, self.dt, self.V_s_tilde,
                                        self.B, self.xx, self.yy, self.u, self.v)
        self.state[:4] = rl_next_state
        
        self.rl_traj.append(rl_next_state)
        self.rl_ctrl.append(self.udir_rl)
        self.state[4:6] = rl_target_direc
        self.state[6] = np.linalg.norm(self.state[:2]-self.target)
         
        Ux = self.bilinear(rl_next_state[0], rl_next_state[1], self.xx, self.yy, self.u)
        Uy = self.bilinear(rl_next_state[0], rl_next_state[1], self.xx, self.yy, self.v)
        omg_xy = self.bilinear(rl_next_state[0], rl_next_state[1], self.xx, self.yy, self.omg)
        omg_xy = np.float(omg_xy)

        self.state[7] = omg_xy
        # self.state[8] = Ux
        # self.state[9] = Uy
        self.state[8] = self.t
        self.reward = self._get_reward()
           
        self.u = self.u_precompute[:,:,self.t]
        self.v = self.v_precompute[:,:,self.t]
        self.omg = self.omg_precompute[:,:,self.t]
        

        return self.state.copy(), self.reward, False, {}

    def reset(self):
        self.u = self.u_precompute[:,:,0]
        self.v = self.v_precompute[:,:,0]
        self.omg = self.omg_precompute[:,:,0]
        self.t = 0
        self.state = np.zeros(9)
        x_rl = np.zeros(4)

        self.reward = 0.0
        self.target = self.get_random_position()
        # self.target = np.array([0.5,0.5])
        
        # x_rl[:2] = self.get_random_position()
        radius = 0.45
        x_rl[:2] = self.get_circular_position(radius)
        # x_rl[:2] = np.array([0.9,0.1])
        x_rl[2:] = self.target_direction(x_rl[:2], self.target)
        
        self.x_naive = x_rl.copy()
        self.udir_rl = x_rl[2:]
        self.udir_naive = self.udir_rl.copy()
        Ux = self.bilinear(x_rl[0], x_rl[1], self.xx, self.yy, self.u)
        Uy = self.bilinear(x_rl[0], x_rl[1], self.xx, self.yy, self.v)
        omg_xy = self.bilinear(x_rl[0], x_rl[1], self.xx, self.yy, self.omg)
        omg_xy = np.float(omg_xy)
        target_direc = self.target_direction(x_rl[:2], self.target)
        self.state[:4] = x_rl.copy()
        self.state[4:6] = target_direc.copy()
        self.state[6] = np.linalg.norm(self.state[:2]-self.target)
        self.state[7] = omg_xy
        self.state[8] = self.t
        # self.state[8] = Ux
        # self.state[9] = Uy
        self.rl_traj = []
        self.naive_traj = []
        self.naive_ctrl = []
        self.rl_ctrl = []
        self.naive_traj.append(self.x_naive)
        self.naive_ctrl.append(self.udir_naive)
        self.rl_traj.append(self.state[:4])
        self.rl_ctrl.append(self.udir_rl)
        return self.state.copy()

    def render(self):
        clev = 15
        plt.clf()
        cont = plt.contourf(self.xx, self.yy, self.omg_precompute[:, :, self.t], clev)
        plt.colorbar()
        # Position of the target
        plt.scatter(self.target[0], self.target[1], color='red')
        # Current position of the RL microswimmer
        plt.scatter(self.state[0], self.state[1], color='blue')
        # Current direction of the RL microswimmer
        plt.quiver(self.state[0], self.state[1], self.state[2], self.state[3], color='blue')
        # Current position of the naive microswimmer
        plt.scatter(self.x_naive[0], self.x_naive[1], color='green')
        # Current direction of the microswimmer
        plt.quiver(self.x_naive[0], self.x_naive[1], self.x_naive[2], self.x_naive[3], color='green')
        # RL Control direction
        plt.quiver(self.state[0], self.state[1], self.udir_rl[0], self.udir_rl[1], color='black')
        # Naive Control direction
        plt.quiver(self.x_naive[0], self.x_naive[1], self.udir_naive[0], self.udir_naive[1], color='black')
    
    def _get_reward(self):
        """
        reward function
        """
        # rew = 128*(np.linalg.norm(self.state.x_adv[:2] - self.target) - np.linalg.norm(self.state.x_rl[:2]-self.target))
        # rew = 128*(np.linalg.norm(self.x_naive[:2] - self.target) - np.linalg.norm(self.state[:2]-self.target))
        # rew = -np.linalg.norm(self.state[:2]-self.target)
        # rew = np.zeros(2)
        a = np.array([0, 0, self.state[6]])
        b = np.array([self.state[2], self.state[3], 0])
        cross_term = 0.5 * np.cross(a, b)
        alignment = np.linalg.norm(cross_term[:2])
        dist = np.linalg.norm(self.x_naive[:2] - self.target) - np.linalg.norm(self.state[:2]-self.target)
        # if self.t < 200:
        #     rew = -alignment + 100*dist
        # else:
        #     rew = -alignment + 100*dist - self.t
        if np.linalg.norm(self.state[:2]-self.target)< 0.05:
            rew = self.state[6]
        else:
            rew = 0.0
        # rew[1] = np.linalg.norm(self.state[:2]-self.target)
        return rew

    def get_actions(self,target_direc):
        rot_45 = np.array([[np.cos(np.pi/4),np.sin(np.pi/4)],[-np.sin(np.pi/4),np.cos(np.pi/4)]])
        rot_90 = np.array([[0, 1], [-1, 0]])
        o_45 = np.dot(rot_45, target_direc)
        o_135 = np.dot(rot_90, o_45)
        o_90 = np.dot(rot_90, target_direc)
        actions = np.stack((target_direc, o_45, o_90,o_135,-target_direc,-o_45,-o_90,-o_135), axis=1)
        return actions

    def bilinear(self,xq, yq, xx, yy, f):
        x = np.array([xx[0, :]])
        y = np.array([yy[:, 0]])
        interp_spline = RectBivariateSpline(y, x, f, kx=1, ky=1)
        fq = interp_spline(yq, xq)
        return fq
    
    def target_direction(self,current_pos, target):
        target_direc = np.zeros((2, 1))
        target_direc = target - current_pos
        target_direc = target_direc / (np.linalg.norm(target_direc)+1e-8)
        return target_direc
    
    def euler_update_position(self,cur_pos, cur_direc, target_direc, omg, dt, V_s_tilde, B, xx, yy, u, v):
        next_pos = np.zeros(4)
        total_vel = np.sqrt(u ** 2 + v ** 2)
        U_rms = np.sqrt(np.mean(total_vel ** 2))
        V_s = V_s_tilde * U_rms
        next_pos[:2] = cur_pos + dt*self.dpos_dt(cur_pos,xx,yy,u,v,cur_direc,V_s)
        next_direc = cur_direc + dt*self.ddirec_dt(cur_pos,cur_direc,omg,B,target_direc,xx,yy)
        next_pos[2:] = next_direc/np.linalg.norm(next_direc)
        return next_pos
    
    def ddirec_dt(self,cur_pos, cur_direc, omg, B, target_direc,xx,yy):
        # Returns rate of change of direction
        omg_xy = self.bilinear(cur_pos[0], cur_pos[1], xx, yy, omg)

        omg_xy = np.float(omg_xy)
        a = np.array([0, 0, omg_xy])
        b = np.array([cur_direc[0], cur_direc[1], 0])
        cross_term = 0.5 * np.cross(a, b)
        d_direc = np.zeros(2)
        c = np.dot(target_direc, cur_direc)
        # c = c/np.linalg.norm(c)
        d_direc = (1 / (2 * B)) * (target_direc - c * cur_direc) + cross_term[:2]
        return d_direc
    
    def dpos_dt(self,cur_pos, xx, yy, u, v, cur_direc, V_s):
        # Returns rate of change of position
        Ux = self.bilinear(cur_pos[0], cur_pos[1], xx, yy, u)
        Ux = np.float(Ux)
        Uy = self.bilinear(cur_pos[0], cur_pos[1], xx, yy, v)
        Uy = np.float(Uy)
        u_xy = np.array([Ux, Uy])
        rate = u_xy + V_s * cur_direc
        return rate
    
  
    def get_circular_position(self,radius):
        theta = np.random.randint(0,359)
        pos = np.zeros(2)
        pos[0] = self.target[0] + radius*np.cos((theta/180)*np.pi)
        pos[1] = self.target[1] + radius*np.sin((theta/180)*np.pi)
        return pos
    
    def set_state_and_target(self, state,target):
        """
        set the field and the next piece
        """
        self.target = target
        self.state = state.copy()
        
        
    def get_random_position(self,lower_bound=10, upper_bound=90):
        p = np.random.randint(lower_bound, upper_bound)
        q = np.random.randint(lower_bound, upper_bound)
        pos = np.array([p / 100, q / 100])
        return pos

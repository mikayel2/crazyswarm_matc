# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 17:36:34 2024

@author: astgh
"""
import casadi as ca
import numpy as np

class MPC:
    def __init__(self, nx=2, nu=1, h=0.1, K=10, trajs=None, du11 = 5, du12 = 2.5, du21 = 0.25, du22 = 0.125, du31 = 0.25, du32 = 0.125, A=np.zeros((2,2)), B=np.zeros((2,1)), agent_idx=0, num_agents=2, cav=False, delta = 1.0):
        self.nx = nx
        self.nu = nu
        self.h = h
        self.A = A
        self.B = B
        self.K = K
        self.agent_idx = agent_idx
        self.num_agents = num_agents
        self.trajs = trajs
        self.du = du11
        self.du2 = du21
        self.du11 = du11
        self.du12 = du12
        self.du21 = du21
        self.du22 = du22
        self.du31 = du31
        self.du32 = du32
        self.x_prev = ca.DM.zeros(self.nu*self.K+self.nx*(self.K+1),1)
        self.x_buffer = []
        self.u_buffer = []
        self.delta = delta  # parameter for path following
        self.cav = cav
        self.setup_MPC()
        
    def F_i2(self, x, gamma_all):
        cost = 0
        gamma_i = x[0]
        for j in range(self.num_agents):
            if j != self.agent_idx:
                dist = ca.norm_2(self.trajs[self.agent_idx].full_state(gamma_i)[0] - self.trajs[j].full_state(gamma_all[j])[0])
                cost += (2*self.agent_idx+1)*(1/dist**2)*self.phi2(dist)
        return cost
    
    def phi3(self, x):
        return ca.if_else(
            x <= self.du2/2,
            0,
            ca.if_else(
                ca.logic_and(x <= self.du2, x >= self.du2/2),
                (2*x - self.du2)**2 / (self.du2**2),
                1
            )
        )  
    
    def phi2(self, x):
        return ca.if_else(
            x <= self.du2/2,
            1,
            ca.if_else(
                ca.logic_and(x <= self.du2, x >= self.du2/2),
                4*(x - self.du2)**2 / (self.du2**2),
                0
            )
        )   
    
    def F_i0(self, x, gamma_all,L): #,neighbors):
        cost = 0
        gamma_i = x[0]

        
        for j in range(self.num_agents):
            if j != self.agent_idx:
                dist = ca.norm_2(self.trajs[self.agent_idx].full_state(gamma_i)[0] - self.trajs[j].full_state(gamma_all[j])[0])
                cost_temp = self.phi(dist)*(gamma_i - gamma_all[j])**2 #self.phi3(dist)*self.phi(dist)*(gamma_i - gamma_all[j])**2
                cost_temp*= L[self.agent_idx, j]
                cost += cost_temp

        
        
        
        # for j in range(self.num_agents):
        #     j_index = neighbors[0,0]
        #     #j_index_k = int(j_index[k].full()[0])  # Convert to integer    
        #     dist = ca.norm_2(self.trajs[self.agent_idx].full_state(gamma_i)[0] - self.trajs[j].full_state(gamma_all[j])[0])
        #     cost += self.phi3(dist)*self.phi(dist)*(gamma_i - gamma_all[j])**2        
        return cost
    
    def phi(self, x):
        return ca.if_else(
            x <= self.du/2,
            1,
            ca.if_else(
                ca.logic_and(x <= self.du, x >= self.du/2),
                4*(x - self.du)**2 / (self.du**2),
                0
            )
        )
    def dist_to_neighb_2(self, x, gamma_all):
        cost = 0
        gamma_i = x[0]
        for j in range(self.num_agents):
            if j != self.agent_idx:
                dist = ca.norm_2(self.trajs[self.agent_idx].full_state(gamma_i)[0] - self.trajs[j].full_state(gamma_all[j])[0])
                cost += dist**2
                
        return cost

    def dist_to_neighb(self, x, gamma_all,L):
        cost = 0
        gamma_i = x[0]
        for j in range(self.num_agents):
            if j != self.agent_idx:
                cost += (gamma_i - gamma_all[j])**2
                cost*= L[self.agent_idx, j]
                
        return cost
    

    def path_following_error(self, x_pf, x_d_dot, delta):
        # take the distance error
        numerator = ca.mtimes(x_d_dot.T, x_pf)
        # Compute the norm of x_d_dot (the Euclidean norm of each column of x_d_dot)
        norm = ca.norm_2(x_d_dot)  # Taking the norm of the distance
        denominator = norm + delta
        alpha_bar = numerator / denominator
        # alpha_bar = ca.sparsify(alpha_bar.T) #squeeze to get a 1D array
        return alpha_bar


    def distance(self, x, actual_position):
        gamma_i = x[0]
        desired_state = self.trajs[self.agent_idx].full_state(gamma_i)[0]
        return desired_state - actual_position


    def dynamics(self, x, u, x_next):
        return x_next - self.A @ x - self.B @ u
        
    def objective(self, x, u, gamma_all,cav, L):
        #print("Neighbors: ",neighbors)
        gamma_dot = x[1]
        # Default
        #obj = (gamma_dot - 1)**2 + self.F_i0(x, gamma_all) + u**2 # works
        #obj = (gamma_dot - 1)**2 + self.F_i0(x, gamma_all) + self.F_i2(x, gamma_all) + u**2 # stops
        obj = (gamma_dot - 1)**2 + self.F_i0(x, gamma_all,L) + u**2 # stops
        if cav:
            obj += self.F_i2(x, gamma_all)  
        return obj
    
    def objective_terminal(self, x, gamma_all,L):
        gamma_dot = x[1]
        obj =  (gamma_dot - 1)**2 + self.dist_to_neighb(x, gamma_all,L)
        #obj = (gamma_dot - 1) ** 2 + self.F_i0(x, gamma_all, L)
        # + self.dist_to_neighb(x, gamma_all)
        return obj
        
    def setup_MPC(self):
        x = ca.SX.sym('x', self.nx, self.K+1)
        u = ca.SX.sym('u', self.nu, self.K)
        x0 = ca.SX.sym('x0', self.nx, 1)
        gamma_all = ca.SX.sym('gamma_all', self.num_agents, self.K+1)
        L = ca.SX.sym('L', self.num_agents, self.num_agents)
        
        const = [x0 - x[:,0]]
        cost = 0
        
        for k in range(self.K):
            const.append(self.dynamics(x[:,k], u[:,k], x[:,k+1]))
            cost += self.h*self.objective(x[:,k], u[:,k], gamma_all[:,k], self.cav, L)
        
        cost += self.objective_terminal(x[:,self.K], gamma_all[:,self.K],L)
        

        # Set up the NLP problem
        nlp = {'x': ca.vertcat(ca.vec(u), ca.vec(x)), 
       	        'f': cost,
                'g': ca.vertcat(*const),
       	        'p': ca.vertcat(x0, ca.vec(gamma_all), ca.vec(L)), 
            }
        opts = {
                'ipopt.print_level': 0, 
                'print_time': 0
                }
        #opts['ipopt.max_iter'] = 50
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        #self.solver = ca.nlpsol('solver', 'bonmin', nlp,  {'print_time': True})
        #self.solver = ca.nlpsol('solver', 'feasiblesqpmethod', nlp,  {'print_time': True})


        #self.solver = ca.nlpsol('solver', 'qpsol', nlp, opts)
        #self.solver = ca.nlpsol('solver', 'blocksqp', nlp, {'print_time': True})
        
        


    def solve(self, x, gamma_all, x_max, x_min, u_max, u_min, t, actual_position,neighbors,L, agent_idx):

        #  path-following error
        #print(neighbors)

        dist = self.distance(x, actual_position)
        x_d_dot = self.trajs[agent_idx].full_state(x[0])[1]  # take 1st index since it is the velocity returned from the trajectory,
        x_d_dot *= x[1]  # take derivative of the trajectory and multiply by gamma_dot for chain rule page 947 of their paper
        alpha_bar = self.path_following_error(dist, x_d_dot, 1)

        # VERSION 1: adding the path-following error only to the current gamma
        x[0] -=  0.05*alpha_bar
        gamma_all[agent_idx, 0] -= alpha_bar * 0.05
    
        # Solve the problem
        sol = self.solver(x0 = self.x_prev,
        		   p = ca.vertcat(x, ca.vec(gamma_all),ca.vec(L)),
        		   lbg = [0]*self.nx*(self.K+1),
        		   ubg = [0]*self.nx*(self.K+1),
                   lbx = u_min*self.K + x_min*(self.K+1),
                   ubx = u_max*self.K + x_max*(self.K+1))
        
        # Extract and return results
        U_opt = np.array(ca.reshape(sol['x'][:self.nu*self.K], self.nu, self.K))
        X_opt = np.array(ca.reshape(sol['x'][self.nu*self.K:], self.nx, self.K+1))
        cost_solver = np.array(sol['f'])
        self.x_prev = ca.vertcat(sol['x'][1:self.nu*self.K], sol['x'][self.nu*(self.K-1)],
                                          sol['x'][self.nu*self.K + self.nx:], sol['x'][-self.nx:])
        self.x_buffer.append(X_opt)
        self.u_buffer.append(U_opt)
        
        return U_opt[:,0], cost_solver
            
        
#def main():

#if __name__ == "__main__":
#    main()         
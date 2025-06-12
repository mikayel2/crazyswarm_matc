#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#
import matplotlib.pyplot as plt
import numpy as np
#from acados_template import latexify_plot
from scipy import linalg

def modify_laplacian(L):
            n = L.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    if i != j:
                        prob = np.random.normal(0.5,0.5)
                        prob = np.clip(prob, 0, 1)
                        value = 1 if prob >= 0.5 else 0

                        # value = int(random.choice([0, 1]))

                        L[i, j] = value
                        L[j, i] = value
                    else:
                        L[i, j] = 0

                off_diag_sum = 0
                for k in range(n):
                    if k != i:
                        off_diag_sum += L[i, k]
                L[i, i] = -off_diag_sum
            return L


def random_communcation_disturbance(number_of_agent,rand = False):
    
    L = -np.ones((number_of_agent, number_of_agent))
    for k in range(number_of_agent):
        L[k, k] = number_of_agent - 1
    
    if rand:
        for _ in range(2):
            while True:
                i, j = np.random.randint(0, number_of_agent, size=2)
                if i != j:
                    break
            #print("Index i: ",i)
            #print("Index j: ",j)
            if L[i, j] !=0 and L[j, i] != 0:
                L[i, j] = 0
                L[j, i] = 0
                L[i, i] = L[i, i] - 1
                L[j, j] = L[j, j] - 1
    
    neighbors = []
    for i in range(number_of_agent):
        agent_neighbors = []
        for k in range(number_of_agent):
            if L[i, k] != 0 and i != k:
                agent_neighbors.append(k)
        while len(agent_neighbors) < number_of_agent - 1:
            agent_neighbors.append(-1)
        neighbors.append(agent_neighbors)
    neighbors = np.array(neighbors)

    #neighbors = np.array(neighbors)
   
    
    return L, neighbors
        

def laplacian_time_varying(p,l_prev,d_min,d_min_2):
    # print("Executing trajectory")
    d = []
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            d.append(linalg.norm(p[i, :] - p[j, :]))

    l = l_prev

    for i in range(len(d)):
        if d[i] < d_min_2:
           l[i] = 1
        elif d[i] >= d_min_2 and d[i] <= d_min:
            l[i] = ((d[i] - d_min)**2)/((d_min_2 - d_min)**2)

    return np.asarray(l)  


def laplacian(n):
    L = -np.ones((n, n))
    for i in range(n):
        L[i, i] = n-1
    return L

def plot_pendulum(shooting_nodes, u_max, U, X_true, X_est=None, Y_measured=None, latexify=False, plt_show=True, X_true_label=None):
    """
    Params:
        shooting_nodes: time values of the discretization
        u_max: maximum absolute value of u
        U: arrray with shape (N_sim-1, nu) or (N_sim, nu)
        X_true: arrray with shape (N_sim, nx)
        X_est: arrray with shape (N_sim-N_mhe, nx)
        Y_measured: array with shape (N_sim, ny)
        latexify: latex style plots
    """

    if latexify:
        latexify_plot()

    WITH_ESTIMATION = X_est is not None and Y_measured is not None

    N_sim = X_true.shape[0]
    nx = X_true.shape[1]

    Tf = shooting_nodes[N_sim-1]
    t = shooting_nodes

    Ts = t[1] - t[0]
    if WITH_ESTIMATION:
        N_mhe = N_sim - X_est.shape[0]
        t_mhe = np.linspace(N_mhe * Ts, Tf, N_sim-N_mhe)

    plt.subplot(nx+1, 1, 1)
    line, = plt.step(t, np.append([U[0]], U))
    if X_true_label is not None:
        line.set_label(X_true_label)
    else:
        line.set_color('r')

    plt.ylabel('$u$')
    plt.xlabel('$t$')
    plt.hlines(u_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
    plt.hlines(-u_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
    plt.ylim([-1.2*u_max, 1.2*u_max])
    plt.xlim(t[0], t[-1])
    plt.grid()


    states_lables = ['$gamma$', '$gamma-dot$', '$L-12$', '$gamma-2$','$G-11$','$G-12$']

    for i in range(nx):
        plt.subplot(nx+1, 1, i+2)
        line, = plt.plot(t, X_true[:, i], label='true')
        if X_true_label is not None:
            line.set_label(X_true_label)

        if WITH_ESTIMATION:
            plt.plot(t_mhe, X_est[:, i], '--', label='estimated')
            plt.plot(t, Y_measured[:, i], 'x', label='measured')

        plt.ylabel(states_lables[i])
        plt.xlabel('$t$')
        plt.grid()
        plt.legend(loc=1)
        plt.xlim(t[0], t[-1])

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)

    if plt_show:
        plt.show()

def main():
    # n = 2  # Example size of the Laplacian matrix
    # L = laplacian(n)
    # print("Laplacian matrix for n = {}:".format(n))
    # print(L)
    L, neighbors = random_communcation_disturbance(4,rand = True)
    print("L:", L)
    print("Neighbors:", neighbors)
if __name__ == "__main__":
    main()
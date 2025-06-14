'''
New to the code? 
Visit "main.pdf" and read the abstract have it by the side of the code. 
+ Read MPC.py first which has initiazlized in it our MPC problem formulation // Have "main.pdf" ready next to it!
+ Problem 3, page 3
+ To understand the trajectory, read circular_traj_cas.py in "./tools/ circular_traj_cas.py".
+ It is possible to change the trajectory if wanted/needed
+ To implement any changes to the way the problem is being solved, add them to MPC.py
'''

from __future__ import print_function
import csv
#from crazyflie_py import Crazyswarm
from pycrazyswarm import *
import numpy as np
import casadi as ca # Importing CasADi for optimization and MPC
# from tools.inf_traj import inf_traj, go_to_initial_position
from casadi_ocp.MPC import MPC
from trajectories.circular_traj_cas import cir_traj
from trajectories.circular_traj_cas2 import cir_traj_2
#from trajectories.wave_traj_cas import build_parallel_waves
from trajectories.bspline_traj_cas import BsplineTrajCas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from config.config_file_reader_casadi import TimeCoordinationConfig
from utils.utils import laplacian_time_varying
from utils.utils import random_communcation_disturbance
from utils.utils import modify_laplacian
from trajectories.sequential_traj_cas import seq_traj_cas
from trajectories.sequential_traj_cas_2 import seq_traj_cas_2
#import time


config = TimeCoordinationConfig()

# Create a 3D plot for trajectories
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize the plot objects for each agent
lines = []
markers = []
for i in range(config.num_agents):
    line, = ax.plot([], [], [], label=f'agent {i+1}')  # Empty plot initially
    marker, = ax.plot([], [], [], 'o')  # Marker for the current position
    lines.append(line)
    markers.append(marker)

ax.set_title('3D Trajectories of Agents')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([0, 1.5])
ax.legend()
plt.ion() 
plt.show(block=False)
plt.pause(0.1)

def executeTrajectory(swarm, timeHelper):
    """
    Executes the trajectory for the given number of agents using Model Predictive Control (MPC).

    Parameters:
    - swarm: The Crazyswarm object that manages the Crazyflies.
    - duration: Total time to run the trajectory.
    - timeHelper: Helper for managing time and synchronization.
    - num_agents: Number of drones.
    - rate: Frequency of updates.
    - log: Whether to log the data or not.
    """
    allcfs = swarm.allcfs
    
    # Initialize circular trajectories for each drone
    nx = config.nx
    nu = config.nu
    K = config.K
    h = config.h
    T = int(config.flight_duration/config.h+1)

    u_min = config.u_min
    u_max = config.u_max
    x_min = config.x_min
    x_max = [np.inf, config.x_max_up]
    du = config.du

    A = np.array([[1,config.h], [0, 1]])  # Example A matrix
    B = np.array([[config.h**2/2], [config.h]])    # Example B matrix

    # trajectories = [seq_traj_cas_2(config.circle_center_coordiantes_21,config.radius_21,config.angular_velocity_21 ,config.init_angle_21, config.rotation_direction_21,config.duration_21),
    #                 seq_traj_cas_2(config.circle_center_coordiantes_22,config.radius_22,config.angular_velocity_22 ,config.init_angle_22, config.rotation_direction_22,config.duration_22)]


    # trajectories = [seq_traj_cas_2(config.circle_center_coordiantes_21,config.radius_21,config.angular_velocity_21 ,config.init_angle_21, config.rotation_direction_21,config.duration_21),
    #                 seq_traj_cas_2(config.circle_center_coordiantes_22,config.radius_22,config.angular_velocity_22 ,config.init_angle_22, config.rotation_direction_22,config.duration_22),
    #                 seq_traj_cas_2(config.circle_center_coordiantes_23,config.radius_23,config.angular_velocity_23 ,config.init_angle_23, config.rotation_direction_23,config.duration_23),
    #                 seq_traj_cas_2(config.circle_center_coordiantes_24,config.radius_24,config.angular_velocity_24 ,config.init_angle_24, config.rotation_direction_24,config.duration_24)]

    
    # trajectories = [  seq_traj_cas(config.circle_center_coordiantes_1,config.radius_1,config.angular_velocity_1 ,config.init_angle_1, config.rotation_direction_1,config.duration_1),
    #                 seq_traj_cas(config.circle_center_coordiantes_2,config.radius_2,config.angular_velocity_2 ,config.init_angle_2, config.rotation_direction_2,config.duration_2),           
    #                cir_traj_2(config.circle_center_coordiantes[2],config.radius[2],config.angular_velocity[2] ,config.init_angle[2], config.rotation_direction[2]),
    #                cir_traj_2(config.circle_center_coordiantes[3],config.radius[3],config.angular_velocity[3] ,config.init_angle[3], config.rotation_direction[3])
    #                ]

    # ---------- CHOOSE CSV-BASED LANES ------------------------------
    CSV_DIR   = "csv"          # folder that contains lane_0.csv … lane_7.csv
    DURATION  = 20.0           # seconds to traverse the 3-m Y-range
    
    trajectories = [
        BsplineTrajCas(f"{CSV_DIR}/lane_{i}.csv", DURATION)
        for i in range(config.num_agents)
    ]
    # ---------------------------------------------------------------

    # trajectories = build_parallel_waves(
    #     n_curves      = config.num_agents,   # 4, 8, etc.
    #     spacing       = 4.0,                 # lane gap (m)
    #     amplitude     = 2.0,                 # peak height
    #     wavelength    = 12.0,                # peak-to-peak distance
    #     forward_speed = 1.5,                 # m s-1 along +x
    #     z0            = 1.0                  # hover altitude
    #     )
    #seq_traj(config)
    # Create circular trajectories for each drone
    #trajectories = []
    
    #for i in range(config.num_agents):
    #    traj = cir_traj_2(config.circle_center_coordiantes[i],config.radius[i],config.angular_velocity[i] ,config.init_angle[i], config.rotation_direction[i])
    #    trajectories.append(traj)
        
    mpcs = [MPC(nx=config.nx, nu=config.nu, h=config.h, K=config.K,
              trajs=trajectories, du11 = config.du11, du12 = config.du12, du21 = config.du21, du22 = config.du22, du31 = config.du31, du32 = config.du32,
              A=A, B=B, agent_idx=i, num_agents=config.num_agents,cav=False) for i in range(config.num_agents)]

    
    
    # Initialize the initial state for each drone
    # x0 = np.vstack([np.array(config.gamma_init[0]),
    #                  np.array(config.gamma_init[1])]).T
    x0 = np.vstack([np.array(config.gamma_init[0]),
                    np.array(config.gamma_init[1]),
                    np.array(config.gamma_init[2]),
                    np.array(config.gamma_init[3])]).T
                    
    ranges = [np.arange(0, x0[0, i], config.h) for i in range(x0.shape[1])]

    # Find the maximum length among the generated arrays
    max_length = max([len(r) for r in ranges])

    # Extend the shorter arrays by repeating the final value to match the length of the longest array
    ranges_extended = []
    for r in ranges:
        if len(r) == 0:
            # If the array is empty, fill it with a constant array of the maximum length
            extended = np.full(max_length, 0)
        else:
            # Otherwise, pad the array by repeating the last element
            extended = np.pad(r, (0, max_length - len(r)), mode='edge')
        ranges_extended.append(extended)

    # Stack the extended arrays as rows of the new matrix
    # new_times = np.vstack(ranges_extended).T
    # t__ = 0
    # while t__ < new_times.shape[0]:
    #   for i, cf in enumerate(swarm.allcfs.crazyflies):
    #           if i >= len(trajectories):
    #               print(f"Error: Index {i} out of range for trajectories list.")
    #               break
    #           traj = trajectories[i]  # Get the correct trajectory object for each drone
    #           # position_, velocity_, acceleration_, yaw_, omega_ = traj.trajectory_function(t/rate)
    #           position, velocity, acceleration, yaw, omega = traj.full_state(new_times[t__,i]) 

    #           # Convert CasADi expressions to numeric values
    #           position = np.array(position).flatten()
    #           velocity = np.array(velocity).flatten()
    #           acceleration = np.array(acceleration).flatten()
    #           yaw = float(yaw)
    #           omega = np.array(omega).flatten()

    #           # Update drone commands with computed values
    #           # line could be removed if using another plotting/ not using crazyswarm library
    #           #cf.cmdFullState(position, velocity, acceleration, yaw, omega)
    #           cf.cmdPosition(position, yaw=0.0)

    #   t__ += 1
    #   timeHelper.sleepForRate(config.rate)
        
    # gamma_all = np.vstack((np.arange(x0[0,0], (config.K+1)*config.h+x0[0,0], config.h),
    #                         np.arange(x0[0,1], (config.K+1)*config.h+x0[0,1], config.h)))
    gamma_all = np.vstack((np.arange(x0[0,0], (config.K+1)*config.h+x0[0,0], config.h),
                           np.arange(x0[0,1], (config.K+1)*config.h+x0[0,1], config.h),
                           np.arange(x0[0,2], (config.K+1)*config.h+x0[0,2], config.h),
                           np.arange(x0[0,3], (config.K+1)*config.h+x0[0,3], config.h)))
    print("gamma_all = ", gamma_all)
    gamma_all_new = gamma_all.copy()

    u = np.zeros((T, config.nu, config.num_agents))
    x = np.zeros((T+1, config.nx, config.num_agents))
    curr_traj = np.zeros((T+1, 3, config.num_agents))
    cost = np.zeros((T, config.num_agents))
    x[0] = x0.copy()

    start_time = timeHelper.time()
    t = 0
    times = []

    data_save ={
         't':np.zeros(1),
         'l': np.zeros(int((config.num_agents**2 - config.num_agents)/2)),
         'x_min': np.zeros(2),
    }
    # Initialize arrays to save actual and desired positions
    actual_positions = np.zeros((1, 3))
    desired_positions = np.zeros((1, 3))

    # Communication disturbance
    t_tack = 0
    state_m = False
    _, neighbors = random_communcation_disturbance(config.num_agents, True)


    # NEW L
    L = np.ones((config.num_agents, config.num_agents))
    #print(L)# Initialize all elements to 1
    np.fill_diagonal(L, -(config.num_agents - 1))
    print(L)

    # Main loop for trajectory execution
    # if not using crazyswarm, remove timeHelper and add True instead
    while not timeHelper.isShutdown():

        t_ = timeHelper.time() - start_time

        if x[t,0,0] >= config.flight_duration and x[t,0,1] >= config.flight_duration and x[t,0,2] >= config.flight_duration and x[t,0,3] >= config.flight_duration:
            break
        if t_ >= config.flight_duration:
            break

        # print(gamma_all)
        # if np.linalg.norm(np.array(allcfs.crazyflies[0].position()) - np.array([0.5,0.0,1.0])) < 0.1 or np.linalg.norm(np.array(allcfs.crazyflies[1].position()) - np.array([0.5,0.0,1.0])) < 0.1 or np.linalg.norm(np.array(allcfs.crazyflies[2].position()) - np.array([0.5,0.0,1.0])) < 0.1:
        #     import pdb; pdb.set_trace()

        t = round(t_*config.rate)
        
        times.append(t_)
        # debugging purposes
        #print("t_: ",t_)
        #print("t: ",t)
        x_min = [0.0, 0.0]  
        gamma_all = gamma_all_new.copy()


        # Random communication disturbance
        if t_ - t_tack >= 5:

            if state_m:
                #L, neighbors = random_communcation_disturbance(config.num_agents, True)
                #L = -L
                print("L: ",L)
                L = modify_laplacian(L)
                # print(f"Step {t}: Laplacian matrix L = \n{L}")
                # eigenvalues = np.linalg.eigvals(L)
                # print(f"Step {t}: Eigenvalues of L = {np.sort(eigenvalues)}")
                state_m = False
                t_tack = t_
            else:    
                #L, neighbors = random_communcation_disturbance(config.num_agents, False)
                #L = -L
                print("L: ",L)
                # #print("neighbors: ",neighbors)
                L = modify_laplacian(L)
                # print(f"Step {t}: Laplacian matrix L = \n{L}")
                # eigenvalues = np.linalg.eigvals(L)
                # print(f"Step {t}: Eigenvalues of L = {np.sort(eigenvalues)}")
                state_m = True
                t_tack = t_



        for i in range(config.num_agents):
            mpc = mpcs[i]
           
            #Compute the position difference between the ith crazyflie and the rest
            for j in range(config.num_agents):
                 if i != j:
                     pos_i = np.asarray(allcfs.crazyflies[i].position())
                     pos_j = np.asarray(allcfs.crazyflies[j].position())
                     distance = np.linalg.norm(pos_i - pos_j)
                     if distance <= config.dupc:
                         x_min = [0.0,1.0]
                     
            
            u[t,:,i], cost[t,i] = mpc.solve(x[t,:,i], gamma_all, x_max, x_min, config.u_max, config.u_min, t,  allcfs.crazyflies[i].position(), neighbors,L,i)

            curr_traj[t,:,i] = np.array(trajectories[i].full_state(x[t,0,i])[0]).squeeze()
            
            x[t+1,:,i] = A @ x[t,:,i] + B @ u[t,:,i]
            
            approx_x = A @ mpc.x_buffer[-1][:,-1]
            gamma_all_new[i,:] = np.hstack([mpc.x_buffer[-1][0, 1:], approx_x[0]])
        
            # Update the line data for the trajectory
            lines[i].set_data(curr_traj[:t+1, 0, i], curr_traj[:t+1, 1, i])
            lines[i].set_3d_properties(curr_traj[:t+1, 2, i])
            
            # Update the marker position
            markers[i].set_data([curr_traj[t, 0, i]], [curr_traj[t, 1, i]])
            markers[i].set_3d_properties([curr_traj[t, 2, i]])

        # plt.draw()
        # plt.pause(0.01)
        # if not using crazyswarm, remove for i, cf in enumerate(swarm.allcfs.crazyflies)
        # instead put
        # for i in range(num_agents)
        data_save['t'] = np.vstack([data_save['t'], [t_]])
        for i, cf in enumerate(swarm.allcfs.crazyflies):
            if i >= len(trajectories):
                print(f"Error: Index {i} out of range for trajectories list.")
                break
            traj = trajectories[i]  # Get the correct trajectory object for each drone
            # position_, velocity_, acceleration_, yaw_, omega_ = traj.trajectory_function(t/rate)
            position, velocity, acceleration, yaw, omega = traj.full_state(x[t,0,i]) 

            # Convert CasADi expressions to numeric values
            position = np.array(position).flatten()
            velocity = np.array(velocity).flatten()
            acceleration = np.array(acceleration).flatten()
            yaw = float(yaw)
            omega = np.array(omega).flatten()

            # Update drone commands with computed values
            # line could be removed if using another plotting/ not using crazyswarm library
            #cf.cmdFullState(position, velocity, acceleration, yaw, omega)
            cf.cmdPosition(position, yaw=0.0)

            position_np = np.zeros((config.num_agents,3))
            for i in range(config.num_agents):
                 position_np[i,:] = np.asarray(allcfs.crazyflies[i].position())

            l  =  laplacian_time_varying(position_np, np.zeros(int((config.num_agents**2 - config.num_agents)/2)), config.du, config.du/2)
            data_save['l'] = np.vstack([data_save['l'], l])
            data_save['x_min'] = np.vstack([data_save['x_min'], x_min])

            # Save actual and desired positions
            actual_positions = np.vstack([actual_positions,allcfs.crazyflies[i].position()])
            desired_positions = np.vstack([desired_positions, position])

            

        


        


        timeHelper.sleepForRate(config.rate)

    # plt.ioff()
    # plt.show(block=True)
    
    
    # 2D plots
    plt.figure(2)
    for i in range(config.num_agents):
        nonzero_indices = np.nonzero(x[:-1,0,i])
        plt.plot(nonzero_indices[0], x[:-1,0,i][nonzero_indices], label=f'agent {i+1}')
    plt.title('Gammas')
    plt.xlabel('time')
    plt.ylabel('gamma')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/Gammas.png')

    plt.figure(3)
    for i in range(config.num_agents):
        nonzero_indices = np.nonzero(x[:-1,1,i])
        plt.plot(nonzero_indices[0],x[:-1,1,i][nonzero_indices], label=f'agent {i+1}')
    plt.title('Gamma Dots')
    plt.xlabel('time')
    plt.ylabel('gamma dot')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/Gamma_Dots.png')

    plt.figure(4)
    for i in range(config.num_agents):
        nonzero_indices = np.nonzero(u[:-1,0,i])
        plt.plot(nonzero_indices[0],u[:-1,0,i][nonzero_indices], label=f'agent {i+1}')
    plt.title('Gamma DotDots')
    plt.xlabel('time')
    plt.ylabel('gamma dot dot')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/Gamma_Dot_Dots.png')

    plt.figure(5)
    for i in range(config.num_agents):
        nonzero_indices = np.nonzero(cost[:,i])
        plt.plot(nonzero_indices[0], cost[:,i][nonzero_indices], label=f'agent {i+1}')
    plt.title('Cost')
    plt.xlabel('time')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/Costs.png')


    # Saving log data to csv files
    if config.log_data:
        with open("log/gamma.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for i in range(x[:,0,0].size):
                writer.writerow(x[i,0,:])
        with open("log/gamma-dot.csv", "w", newline="") as f2:
             writer = csv.writer(f2)
             for i in range(x[:,1,1].size):
                 writer.writerow(x[i,1,:])
        with open("log/gamma-dot-dot.csv", "w", newline="") as f3:
             writer = csv.writer(f3)
             for i in range(u[:,0,0].size):
                 writer.writerow(u[i,0,:])
        with open("log/time.csv", "w", newline="") as f4:
             writer = csv.writer(f4)
             for i in range(len(data_save['t'])):
                 writer.writerow(data_save['t'][i])          
        with open("log/l.csv", "w", newline="") as f5:
             writer = csv.writer(f5)
             for i in range(data_save['l'].shape[0]):
                 writer.writerow(data_save['l'][i])
        with open("log/cost.csv", "w", newline="") as f6:
             writer = csv.writer(f6)
             for i in range(len(cost)):
                 writer.writerow(cost[i,:])         
        with open("log/x_min.csv", "w", newline="") as f7:
                writer = csv.writer(f7)
                for i in range(data_save['x_min'].shape[0]):
                    writer.writerow(data_save['x_min'][i])   
        with open("log/actual_positions.csv", "w", newline="") as f9:
                writer = csv.writer(f9)
                for i in range(actual_positions.shape[0]):
                    writer.writerow(actual_positions[i].flatten())
        with open("log/desired_positions.csv", "w", newline="") as f10:
                writer = csv.writer(f10)
                for i in range(desired_positions.shape[0]):
                    writer.writerow(desired_positions[i].flatten())            

      
# def generate_waypoints(start, target, num_points):
#         return [start + (i / (num_points - 1)) * (target - start) for i in range(num_points)]

def main():
   
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    # cf1 = swarm.allcfs.crazyflies[0] # Selecting paricular crazyfile
    # cf2 = swarm.allcfs.crazyflies[1]
    # cf3 = swarm.allcfs.crazyflies[2]
    # cf4 = swarm.allcfs.crazyflies[3]

    # cf1.setParam("stabilizer/controller", 1) # 1: PID, 2: Mellinger
    # cf1.setParam("posCtlPid/xVelMax", 1.4)
    # cf1.setParam("posCtlPid/yVelMax", 1.4)
    # cf1.setParam("posCtlPid/zVelMax", 1.4)
    # cont = cf1.getParam("stabilizer/controller")
    # max_v_x = cf1.getParam("posCtlPid/xVelMax")
    # max_v_y = cf1.getParam("posCtlPid/yVelMax")
    # max_v_z = cf1.getParam("posCtlPid/zVelMax")
    # print("Contoller Type:", cont)
    # print("Max Vx:", max_v_x)
    # print("Max Vy:", max_v_y)
    # print("Max Vz:", max_v_z)
    # timeHelper.sleep(0.5)

    # cf2.setParam("stabilizer/controller", 1) # 1: PID, 2: Mellinger
    # cf2.setParam("posCtlPid/xVelMax", 1.4)
    # cf2.setParam("posCtlPid/yVelMax", 1.4)
    # cf2.setParam("posCtlPid/zVelMax", 1.4)
    # cont = cf2.getParam("stabilizer/controller")
    # max_v_x = cf2.getParam("posCtlPid/xVelMax")
    # max_v_y = cf2.getParam("posCtlPid/yVelMax")
    # max_v_z = cf2.getParam("posCtlPid/zVelMax")
    # print("Contoller Type:", cont)
    # print("Max Vx:", max_v_x)
    # print("Max Vy:", max_v_y)
    # print("Max Vz:", max_v_z)
    # timeHelper.sleep(0.5)

    # cf3.setParam("stabilizer/controller", 1) # 1: PID, 2: Mellinger
    # cf3.setParam("posCtlPid/xVelMax", 0.9)
    # cf3.setParam("posCtlPid/yVelMax", 0.9)
    # cf3.setParam("posCtlPid/zVelMax", 1.4)
    # cont = cf3.getParam("stabilizer/controller")
    # max_v_x = cf3.getParam("posCtlPid/xVelMax")
    # max_v_y = cf3.getParam("posCtlPid/yVelMax")
    # max_v_z = cf3.getParam("posCtlPid/zVelMax")
    # print("Contoller Type:", cont)
    # print("Max Vx:", max_v_x)
    # print("Max Vy:", max_v_y)
    # print("Max Vz:", max_v_z)
    # timeHelper.sleep(0.5)

    # cf4.setParam("stabilizer/controller", 1) # 1: PID, 2: Mellinger
    # cf4.setParam("posCtlPid/xVelMax", 0.9)
    # cf4.setParam("posCtlPid/yVelMax", 0.9)
    # cf4.setParam("posCtlPid/zVelMax", 1.4)
    # cont = cf4.getParam("stabilizer/controller")
    # max_v_x = cf4.getParam("posCtlPid/xVelMax")
    # max_v_y = cf4.getParam("posCtlPid/yVelMax")
    # max_v_z = cf4.getParam("posCtlPid/zVelMax")
    # print("Contoller Type:", cont)
    # print("Max Vx:", max_v_x)
    # print("Max Vy:", max_v_y)
    # print("Max Vz:", max_v_z)
    # timeHelper.sleep(0.5)

    for cf in allcfs.crazyflies[:config.num_agents]:
        cf.takeoff(targetHeight=config.takeoff_target_height, duration=config.takeoff_duration)
    
    timeHelper.sleep(config.takeoff_duration + config.hover_duration)

    executeTrajectory(swarm, timeHelper)
    print("waiting for landing") 

    for cf in allcfs.crazyflies[:config.num_agents]:
        cf.notifySetpointsStop()
        cf.land(targetHeight=config.land_target_height, duration=config.takeoff_duration)

    timeHelper.sleep(config.sleep_duration_before_shutdown)  

if __name__ == "__main__":
    main()

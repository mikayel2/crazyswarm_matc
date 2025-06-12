# Main high-level interface to the Crazyswarm platform
# from pycrazyswarm import Crazyswarm
from __future__ import print_function
from pycrazyswarm import *
import numpy as np
from scipy import linalg
import csv
import yaml
from ocp_acados.ocp import OCP
from ocp_acados.gamma_model import export_gamma_ode_model
from trajectories.circular_trajectories import circ_traj
from config.config_file_reader_acados import TimeCoordinationConfig
from utils.leastsq_curve_fit import LSQ
from utils.utils import laplacian
from utils.Laplacian import laplacian_timevarying
from utils.utils import laplacian_time_varying




def executeTrajectory(allcfs,timeHelper,config):
    
    # Fixing the offset of the time passed
    start_time = timeHelper.time()

    current_value_data ={
        'gamma': np.array(config.gamma_init),
        'gamma_dot': np.array(config.gamma_dot_init),
        'gamma_ddot': np.zeros(config.number_of_drones),
        'l': np.zeros(int((config.number_of_drones**2 - config.number_of_drones)/2))
    }

    prev_value_data ={
        'gamma': np.array(config.gamma_init),
        'gamma_dot': np.array(config.gamma_dot_init),
        'l': np.zeros(int((config.number_of_drones**2 - config.number_of_drones)/2))
    }
   
    save_data = {
        't': np.array([0.0]),
        'gamma': np.array(config.gamma_init),
        'gamma_dot': np.array(config.gamma_dot_init),
        'gamma_ddot': np.zeros(config.number_of_drones),
        'l': np.zeros(int((config.number_of_drones**2 - config.number_of_drones)/2))
    }

    # Define the Optimal Control Problem
    ocp = OCP(config.Fmax, config.N_horizon, config.Tf, config.compile, config.number_of_drones)   
    ocp_solver, integrator, state_size = ocp.setup() 
    print("State size:",state_size)
    
    # Create circular trajectories for each drone
    trajectories = []
    for i in range(config.number_of_drones):
        traj = circ_traj(config.circle_center_coordiantes[i],config.radius[i],config.angular_velocity[i] ,config.init_angle[i], config.rotation_direction[i])
        trajectories.append(traj)
    
    while not timeHelper.isShutdown(): 
        # Real time passed
        t = timeHelper.time() - start_time  

        # Stoping condition
        if t>=config.flight_duration:
            break
        # TODO: Implement the following code in utils and add static case to the function

        
        position_np = np.zeros((config.number_of_drones,3))
        for i in range(config.number_of_drones):
            position_np[i,:] = np.asarray(allcfs.crazyflies[i].position())
        
        
        l  =  laplacian_time_varying(position_np, prev_value_data['l'], config.d_min, config.d_min_2)
        
        # Laplacian Static
        #l = np.ones(config.number_of_drones)

        # Evaluating state and calculating errors
        alphas = []
        for i in range(config.number_of_drones):
            x, v, acc, yaw, omega = trajectories[i].full_state(prev_value_data['gamma'][i])
            error = -np.asarray(allcfs.crazyflies[i].position()) + x
            v_d = np.asarray(v)
            alpha_bar = np.dot(v_d.T, error) / (linalg.norm(v_d) + 1)
            alphas.append(alpha_bar)
        
        # Solving the Optimal Control Problem
        matrix = np.empty((config.number_of_drones, config.number_of_drones + 1))
        for i in range(config.number_of_drones):
            array = np.arange(config.number_of_drones)
            matrix[i,0] = array[i]
            matrix[i,1] = array[i]
            array = np.delete(array, i)
            matrix[i,2:config.number_of_drones+1] = array[:]
        matrix = matrix.astype(int)
        matrix_I = np.identity(config.number_of_drones)

        for i in range(config.number_of_drones):
            x_0 = np.empty(1)
            x_0[0] = prev_value_data['gamma'][matrix[i,0]]
            #print(x_0)
            x_0 = np.append(x_0, prev_value_data['gamma_dot'][matrix[i,1]])
            #print(x_0)
            x_0 = np.append(x_0, prev_value_data['gamma'][matrix[0,2:config.number_of_drones+1]])
            #print(x_0)
            x_0 = np.append(x_0, prev_value_data['l'])
            #print(x_0)
            x_0 = np.append(x_0, config.b_LSQ[0:config.number_of_drones])
            #print(x_0)
            x_0 = np.append(x_0, matrix_I[i,:])
            #print(x_0)
            x_0 = np.append(x_0, alphas[i])
            #print("Initial state:",x_0.shape)
            simU ,simX = ocp.solve(ocp_solver, integrator, x_0)
            current_value_data['gamma_ddot'][i] = simU[0,0]

    
        # Saving Laplacian values
        l = prev_value_data['l']

        # # Solving Least Square for Laplacian
        # gammas = [np.array([simX_1[0][0], simX_1[1][0]]), np.array([simX_2[0][0], simX_2[1][0]]), np.array([simX_3[0][0], simX_3[1][0]]),  np.array([simX_4[0][0], simX_4[1][0]])]
        # time = np.array([0.0, config.Tf/config.N_horizon])
        # lsq_results = []

        # for gamma in gammas:
        #     lsq = LSQ(time, gamma)
        #     a, b = lsq.optz()
        #     lsq_results.append((b))
        
        # Send command to the drones
        for i in range(config.number_of_drones):
            current_value_data['gamma_dot'][i] = prev_value_data['gamma_dot'][i] + current_value_data['gamma_ddot'][i]*(1/config.rate)
            current_value_data['gamma'][i] = prev_value_data['gamma'][i] + prev_value_data['gamma_dot'][i]*(1/config.rate) + 0.5*current_value_data['gamma_ddot'][i]*(1/config.rate)**2
            prev_value_data['gamma_dot'][i] = current_value_data['gamma_dot'][i]
            prev_value_data['gamma'][i] = current_value_data['gamma'][i]
            x,v,acc,yaw,omega = trajectories[i].full_state(current_value_data['gamma'][i])
            allcfs.crazyflies[i].cmdFullState(x,v,acc,yaw,omega)
            #allcfs.crazyflies[i].cmdPosition(x,yaw) # TODO: Check if this works on the real hardware
            #allcfs.crazyflies[i].cmdVelocityWorld(v,yawRate=0) # TODO: Check if this works on the real hardware

        # Log values of gamma, gamma-dot and gamma-dotdot, x, xd, error 
        if config.log_data:
            save_data['t'] = np.vstack([save_data['t'], [t]])
            save_data['gamma'] = np.vstack([save_data['gamma'], current_value_data['gamma']])
            save_data['gamma_dot'] = np.vstack([save_data['gamma_dot'], current_value_data['gamma_dot']])
            save_data['gamma_ddot'] = np.vstack([save_data['gamma_ddot'], current_value_data['gamma_ddot']])
            save_data['l'] = np.vstack([save_data['l'], l])

        timeHelper.sleepForRate(config.rate)

    # Saving log data to csv files
    if config.log_data:
        with open("log/gamma.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for i in range(save_data['gamma'].shape[0]):
                writer.writerow(save_data['gamma'][i])
        with open("log/gamma-dot.csv", "w", newline="") as f2:
            writer = csv.writer(f2)
            for i in range(save_data['gamma_dot'].shape[0]):
                writer.writerow(save_data['gamma_dot'][i])
        with open("log/gamma-dot-dot.csv", "w", newline="") as f3:
            writer = csv.writer(f3)
            for i in range(save_data['gamma_ddot'].shape[0]):
                writer.writerow(save_data['gamma_ddot'][i])
        with open("log/time.csv", "w", newline="") as f4:
            writer = csv.writer(f4)
            for i in range(save_data['t'].shape[0]):
                writer.writerow(save_data['t'][i])          
        with open("log/l.csv", "w", newline="") as f5:
            writer = csv.writer(f5)
            for i in range(save_data['l'].shape[0]):
                writer.writerow(save_data['l'][i])
    
        
def main():
    swarm = Crazyswarm() # Object containing all the functionality of the Crazyswarm
    timeHelper = swarm.timeHelper # Object containing all time-related functionality
    
    # Read the configuration file
    config = TimeCoordinationConfig()

    # INIT object containing all the drones and takeoff all UAVs
    allcfs = swarm.allcfs 
    allcfs.takeoff(config.takeoff_target_height, duration=config.takeoff_duration) 
    timeHelper.sleep(config.takeoff_duration + config.hover_duration)

    # Run time coordination algorithm 
    executeTrajectory(allcfs,timeHelper,config)


    # Land all drones
    for cf in allcfs.crazyflies:
        cf.notifySetpointsStop() # Notify the drones to stop sending setpoints
        cf.land(targetHeight=config.land_target_height, duration=config.land_duration)
    
    # allcfs.stop() # Stop sending setpoints to the drones # TODO: Check if this works on the real hardware    
    # allcfs.land(targetHeight=config.land_target_height, duration=config.land_duration) # Land the drones

    timeHelper.sleep(config.sleep_duration_before_shutdown) 
    
if __name__ == "__main__":
    main()
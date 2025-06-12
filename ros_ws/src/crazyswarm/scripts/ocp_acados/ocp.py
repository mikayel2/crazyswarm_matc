from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from ocp_acados.gamma_model import export_gamma_ode_model
from utils.utils import plot_pendulum
import numpy as np
import scipy.linalg
from casadi import vertcat, sum1


class OCP:
    def __init__(self, Fmax, N_horizon, Tf, compile=False, number_of_drones=3):

        self.Fmax = Fmax
        self.N_horizon = N_horizon
        self.Tf = Tf
        self.compile = compile
        self.number_of_drones = number_of_drones

    def setup(self):
    
        ocp = AcadosOcp()
        model = export_gamma_ode_model(self.number_of_drones)
        ocp.model = model
        ocp.dims.N = self.N_horizon  
       


        # [x1,      x2,        x3,       x4,    x5,   x6,   x7,   x8, x9,  x10,  x11,  x12,  x13]
        # [gamma, gamma_dot, gamma_2, gamma_3, l_12, l_13, l_23, a_1, a_2, G_11, G_12, G_13, error]

        # Row stting vector
        g_index_start = 2*self.number_of_drones - 1 + int(((self.number_of_drones**2) - self.number_of_drones)/2)
        g_index_end = g_index_start + self.number_of_drones
        G = model.x[g_index_start:g_index_end].T # TODO: Check if this is correct

        # Laplacian matrix
        L = np.zeros((self.number_of_drones, self.number_of_drones))
        L_start_index =  self.number_of_drones + 1
        print("Start index:",L_start_index)
        L_end_index = L_start_index + int(((self.number_of_drones**2) - self.number_of_drones)/2)


        for i in range(self.number_of_drones):
            for j in range(i,self.number_of_drones):
                if i != j:
                    L[i,j] =  L[i,j] + L_start_index
                    L_start_index += 1
        L = (L + L.T).astype(int)
        print("Laplacian matrix:",L)

        # Extract non-zero components of each row of L
        L_non_zero = []
        for i in range(self.number_of_drones):
            row_non_zero =L[i, L[i, :] != 0]
            L_non_zero.append(row_non_zero)
        
        L_non_zero = np.array(L_non_zero, dtype=object)
        print("Non-zero components of L:", L_non_zero)

        
        
        L_t = np.empty((0, self.number_of_drones), dtype=object)
        for i in range(self.number_of_drones):
            row = np.array([[sum1(vertcat(*[model.x[idx] for idx in L_non_zero[i]]))]])

            for j in range(len(L_non_zero[i])):
                row = np.append(row,-model.x[L_non_zero[i,j]])
            L_t = np.vstack([L_t, row])    
        
        gamma = np.array([model.x[0]])
        
        for i in range(2,1 + self.number_of_drones):
            gamma = np.append(gamma, model.x[i])


        # the 'EXTERNAL' cost type can be used to define general cost terms
        # NOTE: This leads to additional (exact) hessian contributions when using GAUSS_NEWTON hessian.
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'
        error_index = 2*self.number_of_drones + int(((self.number_of_drones**2) - self.number_of_drones)/2) + self.number_of_drones
        ocp.model.cost_expr_ext_cost = ((model.x[1] - 1)**2) + ((G @ L_t @ gamma)**2) + (model.u[0]**2) + model.x[error_index] 
        ocp.model.cost_expr_ext_cost_e =  ((model.x[1] - 1)**2) + ((G @ L_t @ gamma)**2) + model.x[error_index]
        
        
        # set constraints
        ocp.constraints.x0 = np.zeros(model.x.shape[0])
        
        # solver settings
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' 
        ocp.solver_options.integrator_type = 'IRK'
        ocp.solver_options.sim_method_newton_iter = 10
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
            
        # set QP solver options
        ocp.solver_options.qp_solver_cond_N = self.N_horizon

        # set prediction horizon
        ocp.solver_options.tf = self.Tf

        # create ocp solver object
        solver_json = 'acados_ocp_' + model.name + '.json'
        acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json, build = self.compile)

        # create an integrator with the same settings as used in the OCP solver.
        acados_integrator = AcadosSimSolver(ocp, json_file = solver_json, build = self.compile)
        return acados_ocp_solver, acados_integrator, model.x.shape[0]
    
    def solve(self, ocp_solver, integrator, x0):

        # set initial condition
        nx = ocp_solver.acados_ocp.dims.nx # number of states
        nu = ocp_solver.acados_ocp.dims.nu # number of inputs

        self.Nsim = self.N_horizon # 100 # simulation length 80 [s]
        self.simX = np.ndarray((self.Nsim+1, nx))
        self.simU = np.ndarray((self.Nsim, nu))

        self.simX[0,:] = x0 # initial state
        t = np.zeros((self.Nsim))

        # do some initial iterations to start with a good initial guess
        num_iter_initial = 5
        for _ in range(num_iter_initial):
            #ocp_solver.
            ocp_solver.solve_for_x0(x0_bar = x0) 

            # closed loop
        #for i in range(self.Nsim):
        i=0
        # solve ocp and get next control input
        self.simU[i,:] = ocp_solver.solve_for_x0(x0_bar = self.simX[i, :])

        t[i] = ocp_solver.get_stats('time_tot')

        # simulate system
        self.simX[i+1, :] = integrator.simulate(x=self.simX[i, :], u=self.simU[i,:])

        # evaluate timings

        # scale to milliseconds
        #t *= 1000
        #self.t = t

        #ocp_solver = None

        return self.simU , self.simX # self.simU

        

    def plot(self):

        # scale to milliseconds
        print(f'Computation time in ms: min {np.min(self.t):.3f} median {np.median(self.t):.3f} max {np.max(self.t):.3f}')
        # plot results
        plot_pendulum(np.linspace(0, (self.Tf/self.N_horizon)*self.Nsim, self.Nsim+1), self.Fmax, self.simU, self.simX)



def main():

    # Create the OCP Solver
    # [x1,      x2,        x3,       x4,    x5,   x6,   x7,   x8, x9,  x10,  x11,  x12,  x13]
    # [gamma, gamma_dot, gamma_2, gamma_3, l_12, l_13, l_23, a_1, a_2, G_11, G_12, G_13, error]
    x0 = np.array([4.5, 9.5, 1.0, 3.0, 0.0 , 1.0, 0.0,0.0,0.0, 0.0, 0.0, 0.0, 0.0]) # initial state
    Fmax = 50 # used only for plotting the results
    Tf = 0.8 #.8 # prediction horizon
    N_horizon = 20 # 40 # number of control intervals - 32 [s]
    ocp= OCP(x0, Fmax, N_horizon, Tf)   
    ocp_solver, integrator = ocp.setup() 

    # Test 
    x0 = np.array([4.5, 9.5, 1.0, 3.0, 0.0 , 1.0, 0.0,0.0,0.0, 0.0, 0.0, 0.0, 0.0]) # initial state
    simU, SimX = ocp.solve(ocp_solver, integrator, x0)
  
    
    print(simU[1,0])


if __name__ == '__main__':
    main()
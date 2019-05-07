# visulaliser
import FVis3

# Own imports
from numpy import pi as pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#from datetime import datetime

import sys

# from engine import stellar_engine,test_engine_1,test_engine_2
# from star_core import my_stellar_core
import Solar_parameters as Sun  # Holding different given parameters for the Sun

# Plotting style
plt.style.use('bmh')
#plt.matplotlib.rc('text', usetex=True)
#mpl.rcParams['figure.figsize'] = (14,8)
mpl.rcdefaults()

class convection2D:
    # ------------------------------------------------------------------------ #
    # ---------------------------- INITIALIZING ------------------------------ #

    # Some physical constants:
    m_u = 1.66053904e-27        # Unit atomic mass [kg]
    k_B = 1.382e-23             # Boltzmann's constant [m^2 kg s^-2 K^-1]
    G   = 6.672e-11             # Gravitational constant [N m^2 kg^-2]
    mu  = 0.61                  # Mean molecular weight
    g_y = G*Sun.M/Sun.R**2      # Gravitational acceleration

    gamma = 5/3                 # Degrees of freedom parameter
    mymu_kB = mu*m_u/k_B         # Common factor used to reduse FLOPS
    nabla_inc = 1e-4            # Pertubation in nabla above adiabatic value
    nabla     = 2/5+nabla_inc   # Temperature gradient for convection

    p = 1e-2           # Variable time step parameter

    def __init__(self,xmax=12e6,nx=300,ymax=4e6,ny=100,initialise=True,perturb=False):
        # Set up computational volume:
        self.xmax = xmax    # range of x [m]
        self.ymax = ymax    # range of y [m]
        self.nx = nx        # nr cells in x
        self.ny = ny        # nr cells is y
        self.y = np.linspace(0,ymax,ny) # Computational volume, y-direction 
        self.x = np.linspace(0,xmax,nx) # Computational volume, x-direction
        self.delta_y = np.abs(self.y[0]-self.y[1])  # size of cell, y-direction
        self.delta_x = np.abs(self.x[0]-self.x[1])  # size of cell, x-direction

        if initialise:
            self.initialise(perturb=perturb)

        self.forced_dt = 0

    def initialise(self,perturb=False):
        """
        Initialise parameters in 2D arrays;
        temperature, pressure, density, internal energy, vertical and horizontal
        velocities w and u.
        @ perturb - if True apply gaussian perturbation in temperature
        """
        # Set up arrays
        self.T   = np.zeros((self.ny,self.nx))
        self.P   = np.zeros((self.ny,self.nx))
        self.rho = np.zeros((self.ny,self.nx))
        self.e   = np.zeros((self.ny,self.nx))
        self.u   = np.zeros((self.ny,self.nx))  # Completely initialized
        self.w   = np.zeros((self.ny,self.nx))  # Completely initialized

        # Initial values:
        beta_0 = Sun.T_photo/self.mymu_kB/self.g_y   # Factor used in P
        if not perturb:
            for j in range(0,self.ny):     # loop vertically and fill variables
                depth_term = self.nabla*(self.y[j]-self.ymax)
                self.T[j,:] = Sun.T_photo - self.mymu_kB*self.g_y*depth_term
                self.P[j,:] = Sun.P_photo*((beta_0-depth_term)/beta_0)**(1/self.nabla)
            self.e = self.P/(self.gamma-1)
            self.rho = self.e*(self.gamma-1)*self.mymu_kB/self.T

        else:   # perturb the temperature and use it to initialize rho
            sigma_x = 1e6
            sigma_y = 1e6
            mean_x = self.xmax/2
            mean_y = self.ymax/2
            xx,yy = np.meshgrid(self.x,self.y)
            perturbation = np.exp(-0.5*((xx-mean_x)**2/sigma_x**2 + (yy-mean_y)**2/sigma_y**2))

            for j in range(0,self.ny):  # Fill T and P as normal
                depth_term = self.nabla*(self.y[j]-self.ymax)
                self.T[j,:] = Sun.T_photo - self.mymu_kB*self.g_y*depth_term
                self.P[j,:] = Sun.P_photo*((beta_0-depth_term)/beta_0)**(1/self.nabla)
            # # Perturbation:
            # mean = [int(self.xmax/2),int(self.ymax/2)]  # Place blob in center
            # # covariance for spherical perturbation based on temp at surface
            # cov = [[Sun.T_photo*0.1,0],[0,Sun.T_photo*0.1]]
            # perturbation = np.random.multivariate_normal(mean,cov,self.T.shape)
            # print(perturbation.shape)
            self.T +=   Sun.T_photo*perturbation   # Adding gaussian perturbation

            # Update e and rho as usual, with perturbed T (in rho)
            self.e = self.P/(self.gamma-1)
            self.rho = self.e*(self.gamma-1)*self.mymu_kB/self.T            

    # ------------------------------------------------------------------------ #
    # ------------------------------- SOLVER --------------------------------- #
    def hydro_solver(self):
        """
        hydrodynamic equations solver
        """
        # Unpack names
        rho = self.rho
        e = self.e 
        w = self.w
        u = self.u
        rhow = rho*w 
        rhou = rho*u
        P = self.P 
        T = self.T

        # Find flow directions:
        flow = self.get_flow_directions() # [u_pos,u_neg,w_pos,w_neg]
        
        ## Find time-differentials of each primary variable:
        # Density:
        cent_ddx_u    = self.get_central_x(u)
        cent_ddy_w    = self.get_central_y(w)
        up_u_ddx_rho  = self.get_upwind_x(rho,flow[0],flow[1])
        up_w_ddy_rho  = self.get_upwind_y(rho,flow[2],flow[3])

        self.ddt_rho  = - rho*(cent_ddx_u+cent_ddy_w) - u*up_u_ddx_rho\
                         - w*up_w_ddy_rho

        # Horizontal momentum:
        up_u_ddx_u    = self.get_upwind_x(u,flow[0],flow[1])
        up_u_ddy_w    = self.get_upwind_y(w,flow[0],flow[1])
        up_u_ddx_rhou = self.get_upwind_x(rhou,flow[0],flow[1])
        up_w_ddy_rhou = self.get_upwind_y(rhou,flow[2],flow[3])
        cent_ddx_P    = self.get_central_x(P)

        self.ddt_rhou = - rhou*(up_u_ddx_u + up_u_ddy_w) - u*up_u_ddx_rhou\
                         - w*up_w_ddy_rhou - cent_ddx_P

        # Vertical momentum:
        up_w_ddy_w    = self.get_upwind_y(w,flow[2],flow[3])
        up_w_ddx_u    = self.get_upwind_x(u,flow[2],flow[3])
        up_w_ddy_rhow = self.get_upwind_y(rhow,flow[2],flow[3])
        up_u_ddx_rhow = self.get_upwind_x(rhow,flow[0],flow[1])
        cent_ddy_P    = self.get_central_y(P)

        self.ddt_rhow = - rhow*(up_w_ddy_w + up_w_ddx_u) - w*(up_w_ddy_rhow)\
                         - u*up_u_ddx_rhow - cent_ddy_P - rho*self.g_y

        # Energy:
        up_u_ddx_e = self.get_upwind_x(e,flow[0],flow[1])
        up_w_ddy_e = self.get_upwind_y(e,flow[2],flow[3])

        self.ddt_e = - e*(cent_ddx_u+cent_ddy_w) - u*up_u_ddx_e - w*up_w_ddy_e\
                      - P*(cent_ddx_u+cent_ddy_w)

        # Find optimal dt and evolve primary variables
        dt = self.get_timestep()
        self.rho[:,:] = rho + self.ddt_rho*dt
        self.e[:,:]   = e + self.ddt_e*dt
        self.u[:,:]   = (rhou+self.ddt_rhou*dt)/rho
        self.w[:,:]   = (rhow+self.ddt_rhow*dt)/rho

        # Apply boundary conditions before calculating temperature and pressure
        self.set_boundary_conditions()
        self.P[:,:] = (self.gamma-1)*e 
        self.T[:,:] = (self.gamma-1)*self.mymu_kB*e/rho

        return dt
    # ------------------------------------------------------------------------ #
    # ------------------------ GETTERS & SETTERS ----------------------------- #

    def get_timestep(self):
        """
        Get optimal timestep based on max relative change in primary variables
        If dt is too low or too high force it to a min/max value
        """     
        # Max relative changes in each variable:
        max_rel_rho = np.nanmax(np.abs(self.ddt_rho/self.rho))
        max_rel_e   = np.nanmax(np.abs(self.ddt_e/self.e))
        max_rel_x   = np.nanmax(np.abs(self.u/self.delta_x))
        max_rel_y   = np.nanmax(np.abs(self.w/self.delta_y))

        # Max relative change of all variables:
        # Included 1e-5 to remove cases where delta = 0
        delta = np.nanmax(np.array([max_rel_rho,max_rel_e,max_rel_x,max_rel_y,1e-5]))

        dt = self.p/delta   # Optimal dt based on max relative change

        # Forced min/max dt, track nr of forced cases
        if dt<1e-8:
            dt = 1e-8
            self.forced_dt += 1
        elif dt>0.1:
            dt = 0.1
            self.forced_dt += 1

        return dt

    def set_boundary_conditions(self):
        """
        Set vertical boundary conditions for energy, density and velocity
        """
        self.e[0,:]  = (4*self.e[1,:]-self.e[2,:])\
                        /(3-self.g_y*2*self.delta_y*self.mymu_kB/self.T[0,:])
        self.e[-1,:] = (4*self.e[-2,:]-self.e[-3,:])\
                        /(3+self.g_y*2*self.delta_y*self.mymu_kB/self.T[-1,:])

        self.rho[0,:]  = (self.gamma-1)*self.mymu_kB*self.e[0,:]/self.T[0,:]
        self.rho[-1,:] = (self.gamma-1)*self.mymu_kB*self.e[-1,:]/self.T[-1,:]

        self.w[0,:]  = 0
        self.w[-1,:] = 0

        self.u[0,:]  = (4*self.u[1,:]-self.u[2,:])/3
        self.u[-1,:] = (4*self.u[-2,:]-self.u[-3,:])/3

    def get_flow_directions(self):
        """
        Calculates flow directions
        Returns four 2D boolean arrays with indices for the direction of the flow,
        to be used in upwind differencing.
        """
        u_pos = self.u>=0    # Boolean array for positive horizontal flow
        u_neg = self.u<0     # Negative flow

        # Same for the vertical:
        w_pos = self.w>=0
        w_neg = self.w<0

        return u_pos,u_neg,w_pos,w_neg

    def get_central_x(self,var):
        """ 
        Central difference scheme in x-direction with periodic horizontal boundary
        @ var - variable to differentiate
        """
        return (np.roll(var,-1,axis=1)-np.roll(var,1,axis=1))/(2*self.delta_x)

    def get_central_y(self,var):
        """ 
        Central difference scheme in y-direction with periodic vertical boundary
        (vertical boundary is controlled in self.set_boundary_conditions())
        @ var - variable to differentiate
        """
        return (np.roll(var,-1,axis=0)-np.roll(var,1,axis=0))/(2*self.delta_y)

    def get_upwind_x(self,var,pos_id,neg_id):
        """
        Upwind difference scheme in x-direction with periodic horizontal boundary
        @ var - variable to differentiate
        @ pos_id - indices for positive upwind differencing (positive flow)
        @ neg_id - indices for negative upwind differencing (negative flow)
        
        Uses different expressions for the differential based on the sign of flow
        Returns the resulting differential 2D array
        """

        res = np.zeros((self.ny,self.nx))   # resulting array with differential

        diff_pos = (var-np.roll(var,1,axis=1))/self.delta_x  # if positive flow
        diff_neg = (np.roll(var,-1,axis=1)-var)/self.delta_x # if negative flow

        # Filling resulting array with appropiate differentials
        res[pos_id] = diff_pos[pos_id]
        res[neg_id] = diff_neg[neg_id]

        return res

    def get_upwind_y(self,var,pos_id,neg_id):
        """
        Upwind difference scheme in y-direction with periodic vertical boundary
        @ var - variable to differentiate
        @ pos_id - indices for positive upwind differencing (positive flow)
        @ neg_id - indices for negative upwind differencing (negative flow)
        
        Uses different expressions for the differential based on the sign of flow
        Returns the resulting differential 2D array
        """

        res = np.zeros((self.ny,self.nx))   # resulting array with differential

        diff_pos = (var-np.roll(var,1,axis=0))/self.delta_y  # if positive flow
        diff_neg = (np.roll(var,-1,axis=0)-var)/self.delta_y # if negative flow

        # Filling resulting array with appropiate differentials
        res[pos_id] = diff_pos[pos_id]
        res[neg_id] = diff_neg[neg_id]

        return res


    # ------------------------------------------------------------------------ #
    # ------------------------------- SANITY --------------------------------- #
    def sanity_initial_conditions(self):
        """ Simple sanity test to check the initial vertical gradients """

        T1D = self.T[:,int(self.nx/2)]
        P1D = self.P[:,int(self.nx/2)]
        e1D = self.e[:,int(self.nx/2)]
        rho1D = self.rho[:,int(self.nx/2)]
        y = self.y

        fig, ax = plt.subplots(2,2,sharey='all')
        ax[0,0].set_ylabel(r'$y [m]$')
        ax[1,0].set_ylabel(r'$y [m]$')

        ax[0,0].set_title('Temperature')
        ax[0,0].set_xlabel(r'$T [K]$')
        ax[0,0].plot(T1D,y)

        ax[0,1].set_title('Pressure')
        ax[0,1].set_xlabel(r'$P [Pa]$')
        ax[0,1].plot(P1D,y)

        ax[1,0].set_title('Internal energy')
        ax[1,0].set_xlabel(r'$e [J m^{-3}]$')
        ax[1,0].plot(e1D,y)

        ax[1,1].set_title('Density')
        ax[1,1].set_xlabel(r'$\rho [kg m^{-3}]$')
        ax[1,1].plot(rho1D,y)

if __name__ == '__main__':
    BoX = convection2D()
    viz = FVis3.FluidVisualiser()

    if 'init_sanity' in sys.argv:
        # Quick check used during implementation of initial conditions
        BoX.sanity_initial_conditions()

    if 'sanity' in sys.argv:
        viz.save_data(60,BoX.hydro_solver,rho=BoX.rho,e=BoX.e,u=BoX.u,w=BoX.w,\
                        P=BoX.P,T=BoX.T,sim_fps=1.0)

        if 'save' in sys.argv[-2:]:
            viz.animate_2D('T',save=True,video_name='sanity_T')
            viz.animate_2D('P',save=True,video_name='sanity_P')
            viz.animate_2D('rho',save=True,video_name='sanity_rho')
        else:
            viz.animate_2D('w')
        
        viz.delete_current_data()

    if 'viz' in sys.argv:
        BoX.initialise(perturb=True)
        viz.save_data(300,BoX.hydro_solver,rho=BoX.rho,e=BoX.e,u=BoX.u,w=BoX.w,\
                        P=BoX.P,T=BoX.T,sim_fps=0.2)
        #viz.animate_2D('w',save=True)
        viz.animate_2D('T',save=True)
        
        viz.delete_current_data()
        print(BoX.forced_dt)

    if 'show' in sys.argv[-1]:
        plt.show()  
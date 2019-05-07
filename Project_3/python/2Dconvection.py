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
mpl.rcParams['figure.figsize'] = (14,8)
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
    factor = mu*m_u/k_B         # Common factor used to reduse FLOPS
    nabla_inc = 1e-4            # Pertubation in nabla above adiabatic value
    nabla     = 2/5+nabla_inc   # Temperature gradient for convection

    p = 1e-2           # Variable time step parameter


    def __init__(self,xmax=12e6,nx=300,ymax=4e6,ny=100,initialise=True):
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
            self.initialise()

        self.forced_dt = 0

    def initialise(self):
        """
        Initialise main parameters in 2D arrays:
        temperature, pressure, density and internal energy
        also vertical and horizontal velocities w and u
        """
        # Set up arrays
        self.T   = np.zeros((self.ny,self.nx))
        self.P   = np.zeros((self.ny,self.nx))
        self.rho = np.zeros((self.ny,self.nx))
        self.e   = np.zeros((self.ny,self.nx))
        self.u   = np.zeros((self.ny,self.nx))
        self.w   = np.zeros((self.ny,self.nx))
        #self.w[int(self.ny/2),:] = 1e3

        ## Initial values:
        beta_0 = Sun.T_photo/self.factor/self.g_y
        for j in range(0,self.ny):     # loop vertically
            depth_term = self.nabla*(self.y[j]-self.ymax)
            self.T[j,:] = Sun.T_photo - self.factor*self.g_y*depth_term
            self.P[j,:] = Sun.P_photo*((beta_0-depth_term)/beta_0)**(1/self.nabla)
            self.e[j,:] = self.P[j,:]/(self.gamma-1)
            self.rho[j,:] = self.e[j,:]*(self.gamma-1)*self.factor/self.T[j,:]

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

        # Find flow directions:
        flow = self.get_flow_directions() # [u_pos,u_neg,w_pos,w_neg]
        
        ## Find time-differentials of each primary variable:
        # Density:
        dudx_cent    = self.get_central_x(self.u)
        dwdy_cent    = self.get_central_y(self.w)
        drhodx_up_u  = self.get_upwind_x(self.rho,flow[0],flow[1])
        drhody_up_w  = self.get_upwind_y(self.rho,flow[2],flow[3])

        self.drhodt  = - self.rho*(dudx_cent+dwdy_cent) - self.u*drhodx_up_u\
                    - self.w*drhody_up_w

        # Horizontal momentum:
        dudx_up_u    = self.get_upwind_x(self.u,flow[0],flow[1])
        dwdy_up_u    = self.get_upwind_y(self.w,flow[0],flow[1])
        drhoudx_up_u = self.get_upwind_x(self.rho*self.u,flow[0],flow[1])
        drhoudy_up_w = self.get_upwind_y(self.rho*self.u,flow[2],flow[3])
        dPdx_cent    = self.get_central_x(self.P)

        self.drhoudt = - self.rho*self.u*(dudx_up_u + dwdy_up_u) - self.u*drhoudx_up_u\
                    - self.w*drhoudy_up_w - dPdx_cent

        # Vertical momentum:
        dwdy_up_w    = self.get_upwind_y(self.w,flow[2],flow[3])
        dudx_up_w    = self.get_upwind_x(self.u,flow[2],flow[3])
        drhowdy_up_w = self.get_upwind_y(self.w*self.rho,flow[2],flow[3])
        drhowdx_up_u = self.get_upwind_x(self.w*self.rho,flow[0],flow[1])
        dPdy_cent    = self.get_central_y(self.P)

        self.drhowdt = - self.rho*self.w*(dwdy_up_w + dudx_up_w) - self.w*(drhowdy_up_w)\
                    - self.u*drhowdx_up_u - dPdy_cent - self.rho*self.g_y

        # Energy:
        dedx_up_u = self.get_upwind_x(self.e,flow[0],flow[1])
        dedy_up_w = self.get_upwind_y(self.e,flow[2],flow[3])

        self.dedt = - self.e*(dudx_cent+dwdy_cent) - self.u*dedx_up_u - self.w*dedy_up_w\
                 - self.P*(dudx_cent+dwdy_cent)

        # Find optimal dt and evolve primary variables
        dt = self.get_timestep()
        # print('\n')
        # print(np.argmin(np.abs(self.rho)))
        # print('\n')
        self.rho = self.rho + self.drhodt*dt
        # print('\n')
        # print(np.argmin(self.rho),self.rho.shape)
        # print('\n')
        self.e   = self.e + self.dedt*dt
        self.u   = (rhou+self.drhoudt*dt)/self.rho
        self.w   = (rhow+self.drhowdt*dt)/self.rho

        # Apply boundary conditions before calculating temperature and pressure
        self.set_boundary_conditions()
        self.P = (self.gamma-1)*self.e 
        self.T = (self.gamma-1)*self.factor*self.e/self.rho

        #print('\n max w {:3f} \n'.format(np.max(np.abs(self.w))))

        return dt
    # ------------------------------------------------------------------------ #
    # ------------------------ GETTERS & SETTERS ----------------------------- #

    def get_timestep(self):
        """
        return optimal timestep
        """     
        max_rel_rho = np.nanmax(np.abs(self.drhodt/self.rho))
        max_rel_e   = np.nanmax(np.abs(self.dedt/self.e))
        max_rel_x   = np.nanmax(np.abs(self.u/self.delta_x))
        max_rel_y   = np.nanmax(np.abs(self.w/self.delta_y))

        delta = np.nanmax(np.array([max_rel_rho,max_rel_e,max_rel_x,max_rel_y]))#,1e-5)
        #print(delta)

        dt = self.p/delta

        if dt<1e-8:
            dt = 1e-8
            self.forced_dt += 1
        elif dt>0.1:
            dt = 0.1
            self.forced_dt += 1
        # print('\n')
        # print(dt)
        return dt

    def set_boundary_conditions(self):
        """
        set boundary conditions for energy, density and velocity
        """
        self.e[0,:]  = (4*self.e[1,:]-self.e[2,:])/(3-self.g_y*2*self.delta_y*self.factor/self.T[0,:])
        self.e[-1,:] = (4*self.e[-2,:]-self.e[-3,:])/(3+self.g_y*2*self.delta_y*self.factor/self.T[-1,:])

        self.rho[0,:]  = (self.gamma-1)*self.factor*self.e[0,:]/self.T[0,:]
        self.rho[-1,:] = (self.gamma-1)*self.factor*self.e[-1,:]/self.T[-1,:]

        self.w[0,:]  = 0
        self.w[-1,:] = 0

        self.u[0,:]  = (4*self.u[1,:]-self.u[2,:])/3
        self.u[-1,:] = (4*self.u[-2,:]-self.u[-3,:])/3

    def get_flow_directions(self):
        """
        Calculates flow directions
        Returns four 2D boolean arrays with the direction of the flow,
        to be used in upwind differencing.
        """
        u_pos = self.u>=0    # Boolean array for positive horizontal flow
        u_neg = self.u<0     # Negative flow

        # Vertical:
        w_pos = self.w>=0
        w_neg = self.w<0

        return u_pos,u_neg,w_pos,w_neg

    def get_central_x(self,var):
        """ central difference scheme in x-direction on variable var """
        return (np.roll(var,-1,axis=1)-np.roll(var,1,axis=1))/(2*self.delta_x)

    def get_central_y(self,var):
        """ central difference scheme in y-direction on variable var """
        return (np.roll(var,-1,axis=0)-np.roll(var,1,axis=0))/(2*self.delta_y)

    def get_upwind_x(self,var,pos_id,neg_id):
        """
        Upwind difference scheme in x-direction
        @ var - variable to differentiate
        @ pos_id - indices for positive upwind differencing (positive flow)
        @ neg_id - indices for negative upwind differencing (negative flow)
        
        Uses different expressions for the differential based on the sign of flow
        Returns the resulting differential 2D array
        """

        res = np.zeros((self.ny,self.nx))   # resulting array with differential

        diff_pos = (var-np.roll(var,1,axis=1))/self.delta_x # if positive flow
        diff_neg = (np.roll(var,-1,axis=1)-var)/self.delta_x # if negative flow

        # Filling resulting array with appropiate differentials
        res[pos_id] = diff_pos[pos_id]
        res[neg_id] = diff_neg[neg_id]

        return res

    def get_upwind_y(self,var,pos_id,neg_id):
        """
        Upwind difference scheme in y-direction
        @ var - variable to differentiate
        @ pos_id - indices for positive upwind differencing (positive flow)
        @ neg_id - indices for negative upwind differencing (negative flow)
        
        Uses different expressions for the differential based on the sign of flow
        Returns the resulting differential 2D array
        """

        res = np.zeros((self.ny,self.nx))   # resulting array with differential

        diff_pos = (var-np.roll(var,1,axis=0))/self.delta_y # if positive flow
        diff_neg = (np.roll(var,-1,axis=0)-var)/self.delta_y # if negative flow

        # Filling resulting array with appropiate differentials
        res[pos_id] = diff_pos[pos_id]
        res[neg_id] = diff_neg[neg_id]

        return res


    # ------------------------------------------------------------------------ #
    # ------------------------------- SANITY --------------------------------- #
    def sanity_initial_conditions(self):
        """ Simple sanity test to check the initial vertical gradients """

        T1D = self.T[:,0]
        P1D = self.P[:,0]
        e1D = self.e[:,0]
        rho1D = self.rho[:,0]
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

        plt.show()

if __name__ == '__main__':
    test = convection2D()
    viz = FVis3.FluidVisualiser()

    if 'sanity' in sys.argv:
        test.sanity_initial_conditions()

    if 'viz' in sys.argv:
        viz.save_data(120,test.hydro_solver,rho=test.rho,e=test.e,u=test.u,w=test.w,\
                        P=test.P,T=test.T,sim_fps=1.0)
        viz.animate_2D('w')
        viz.animate_2D('T')
        viz.animate_2D('rho')
        viz.delete_current_data()
        print(test.forced_dt)

    plt.show()  
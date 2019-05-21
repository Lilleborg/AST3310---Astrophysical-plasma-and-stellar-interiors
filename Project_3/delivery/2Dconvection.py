# visulaliser
import FVis3

# Own imports
from numpy import pi as pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import ceil

import sys

import Solar_parameters as Sun  # Holding different given parameters for the Sun

# Reset my own matplotlib settings:
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
    mymu_kB = mu*m_u/k_B        # Common factor used to reduse number of FLOPS
    nabla_inc = 1e-4            # Pertubation in nabla above adiabatic value
    nabla     = 2/5+nabla_inc   # Temperature gradient for convection

    p = 1e-2           # Variable time step parameter

    def __init__(self,xmax=12e6,nx=300,ymax=4e6,ny=100,initialise=True,perturb=False):
        # Set up computational volume:
        self.xmax = xmax    # range of x [m]
        self.ymax = ymax    # range of y [m]
        self.nx = nx        # nr cells in x
        self.ny = ny        # nr cells is y
        # 1D vertical array and size of cells, y-direction:
        self.y,self.delta_y = np.linspace(0,ymax,ny,retstep=True)
        # Same in horizontal direction:
        self.x,self.delta_x = np.linspace(0,xmax,nx,retstep=True)
        if initialise:
            self.initialise(perturb=perturb)

        self.forced_dt = 0  # Used to tracked number of forced time steps
        
    def initialise(self,perturb=False,nr=1):
        """
        Initialise parameters in 2D arrays;
        T - temperature,
        P - pressure,
        rho - density,
        e - internal energy,
        u & w - vertical & horizontal velocities.

        @ perturb - if True apply gaussian perturbation in temperature
        @ nr - number of perturbations to apply along x-axis
        """
        # Set up arrays
        self.T   = np.zeros((self.ny,self.nx))
        self.P   = np.zeros((self.ny,self.nx))
        self.rho = np.zeros((self.ny,self.nx))
        self.e   = np.zeros((self.ny,self.nx))
        self.u   = np.zeros((self.ny,self.nx))  # Completely initialized
        self.w   = np.zeros((self.ny,self.nx))  # Completely initialized

        # Initial values:
        beta_0 = Sun.T_photo/self.mymu_kB/self.g_y   # Substitution used in P    
        for j in range(0,self.ny):     # loop vertically and fill variables
            depth_term = self.nabla*(self.y[j]-self.ymax)
            self.T[j,:] = Sun.T_photo - self.mymu_kB*self.g_y*depth_term
            self.P[j,:] = Sun.P_photo*((beta_0-depth_term)/beta_0)**(1/self.nabla)

        if perturb: # Perturb initial temerature
            self.set_perturbation(nr=nr)
        # Using pressure and (possibly perturbed) temperature to init e and rho
        self.e = self.P/(self.gamma-1)
        self.rho = self.e*(self.gamma-1)*self.mymu_kB/self.T    
    # ------------------------------------------------------------------------ #
    # ------------------------------- SOLVER --------------------------------- #
    def hydro_solver(self):
        """
        hydrodynamic equations solver
        """
        # Unpack names for easier reading and compact lines
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
        self.u[:,:]   = (rhou+self.ddt_rhou*dt)/self.rho[:,:]
        self.w[:,:]   = (rhow+self.ddt_rhow*dt)/self.rho[:,:]
        # Note that the flow uses density from last step in variables rhou and
        # rhow, and the density in the next step in self.rho[:,:]

        # Apply boundary conditions before calculating temperature and pressure
        self.set_boundary_conditions()
        self.P[:,:] = (self.gamma-1)*e 
        self.T[:,:] = (self.gamma-1)*self.mymu_kB*e/self.rho[:,:]

        return dt
    # ------------------------------------------------------------------------ #
    # ------------------------ GETTERS & SETTERS ----------------------------- #
    def set_perturbation(self,nr=1):
        """
        Create gaussian perturbations in the initial temperature distribution
        @ nr - number of perturbation "blobs" (experimental above default nr = 1)
        """
        nr *= 2
        x_position = np.arange(1,nr,2)/nr     # x position of blobs
        alt_sign = np.ones(ceil(nr/2)) # array with alternating signs
                                       # to get neg and pos perturbations
        if nr != 2:                    # only if more than one blob
            alt_sign[0::2] = -1
        
        sigma_x = 1e6   # Equal standard deviations for circular blobs
        sigma_y = 1e6
        mean_y = self.ymax/2    # Place blobs in the midle vertically
        xx,yy = np.meshgrid(self.x,self.y)  # simple mesh of computational volume

        for i,scale in enumerate(x_position): # loop over nr of blobs and perturb T
            mean_x = self.xmax*scale          # place blob according to nr
            perturbation = np.exp(-0.5*((xx-mean_x)**2/sigma_x**2\
                            +(yy-mean_y)**2/sigma_y**2))
            # Add the perturbation to the initial temperature,
            # with amplitude based on the the surface temperature,
            # and alternating sign if more than one blob:
            self.T += 0.5*Sun.T_photo*perturbation*alt_sign[i]

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
        if dt<1e-7: # Too low dt causes unstability and prolongs the calculations
            dt = 1e-7
            self.forced_dt += 1
        elif dt>0.1: # Too high dt causes even more unstability 
            dt = 0.1 # (mostly in use in the equilibrium situation)
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
        (vertical boundary is controlled in self.set_boundary_conditions())
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
        # Dictionary with matplotlib configurations
        rcconfig = {'font.size'             : 16.0,
                    'text.usetex'           : True,
                    'text.latex.preamble'   : ('\\usepackage{physics}','\\usepackage{siunitx}'),
                    'figure.figsize'        : (14,8),
                    'figure.constrained_layout.use'     :True,
                    'figure.constrained_layout.h_pad'   :0.07,
                    'figure.constrained_layout.w_pad'   :0.05,
                    'figure.titlesize'      : 'xx-Large',
                    'path.simplify'         : True}
        with plt.rc_context(rc = rcconfig):
            plt.style.use('bmh')
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
            plt.show()
            plt.close('all')

if __name__ == '__main__':
    BoX = convection2D()
    viz = FVis3.FluidVisualiser(fontsize=17)

    # In the following are a quick and dirty way of making the code do different
    # operations or actions utilizing command line arguments:

    # Setup simulation configurations:
    extent = [0,BoX.xmax/1e6,0,BoX.ymax/1e6] # extent of axis in [Mm]
    if len(sys.argv)<3:
        print('No/not enough cmd arguments!')
        if len(sys.argv)<2:
            print('Using basic animation configurations')
            sys.argv.append('basic')
        print('No specified action, defaults to animating T')
        sys.argv.append('animate')

    if 'viz' in sys.argv:  # experiment with simulation configs
        nr = 5
        endtime = 450
        perturb = True
        sim_fps = 0.8
        name_app = str(nr)+'_blobs_endtime_'+str(endtime)+'_'
        folder = 'viz_'+str(nr)+'_blobs_endtime_'+str(endtime)
        subtitle = str(nr)+' alternating perturbations over '\
                +str(endtime)+' seconds'

    elif 'sanity' in sys.argv:  # sanity config
        nr = 1
        endtime = 60
        perturb = False
        sim_fps = 1.0
        name_app = 'sanity_endtime_'+str(endtime)+'_'
        folder = 'sanity_'+str(endtime)
        subtitle = 'hydrostatic equilibrium over '\
                +str(endtime)+' seconds'

    elif 'basic' in sys.argv: # basic simulation config (most simple solution)
        nr = 1 
        endtime = 400
        perturb = True
        sim_fps = 1.0
        name_app = 'basic_endtime_'+str(endtime)+'_'
        folder = 'basic_endtime_'+str(endtime)
        subtitle = 'single positive perturbation over '\
                +str(endtime)+' seconds'

    # Run simulation, or load data
    if not 'load' in sys.argv:  # unless given load cmd, run save data method
        BoX.initialise(perturb=perturb,nr=nr)
        viz.save_data(endtime,BoX.hydro_solver,rho=BoX.rho,e=BoX.e,u=BoX.u,\
                        w=BoX.w,P=BoX.P,T=BoX.T,sim_fps=sim_fps,folder=folder)
        print('Forced number of dt', BoX.forced_dt)
    else:
        folder = input('Which folder to load data from? ')

    # Actions on simulation data; snapshots, animation, init_sanity...

    if 'snap' in sys.argv: # take 20 snap shots of the animation
        snap_time = np.linspace(0,endtime,20,endpoint=False)
        if 'flux' in sys.argv:
            title = 'Horizontally averaged energy flux,\n'+subtitle
            viz.animate_energyflux(snapshots=snap_time,title=title,folder=folder,\
                                    units={'Lx':'Mm','Lz':'Mm'},extent=extent)
        else:
            title = 'Snapshots of T,\n'+subtitle
            viz.animate_2D('T',snapshots=snap_time,title=title,folder=folder,\
                            units={'Lx':'Mm','Lz':'Mm'},extent=extent,height=8)
        plt.close('all')

    if 'init' in sys.argv:
        # Quick check used during implementation of initial conditions
        BoX.sanity_initial_conditions()

    if 'animate' in sys.argv:  # 2D animation primary variables, default only T
        vid_time = 10          # [s]
        vid_fps = endtime*sim_fps/vid_time

        if not 'all' in sys.argv:   # if all as cmd plot for EACH variable
            var_names = ['T']
            if 'sanity' in sys.argv: # also add w if sanity run
                var_names.append('w')
        else:
            var_names = ['T','P','rho','e','u','w'] # all variable names
        
        if 'flux' in sys.argv:  # Animate horizontally averaged energy flux
            
            title = 'Horizontally averaged energy flux,\n'+subtitle
            viz.animate_energyflux(save=True,video_name=name_app+'flux',title=title,\
                                    video_fps=vid_fps,units={'Lx':'Mm','Lz':'Mm'},\
                                    folder=folder,extent=extent)

        else:   # If not flux, animate primary variables
            for var in var_names:
                title = 'Animated '+var+', '+subtitle
                viz.animate_2D(var,save=True,video_name=name_app+var,title=title,\
                                video_fps=vid_fps,units={'Lx':'Mm','Lz':'Mm'},\
                                folder=folder,extent=extent,cbar_aspect=0.05)

    if not 'no_del' in sys.argv:
        viz.delete_current_data()   
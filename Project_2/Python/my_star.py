from numpy import pi as pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import sys
sys.path.append("../../project_0")         # for own use when developing
sys.path.append("../../project_1/python/") # for own use when developing

from engine import stellar_engine,test_engine_1,test_engine_2
from star_core import my_stellar_core
import Solar_parameters as Sun  # Holding different given parameters for the Sun

# Plotting style
plt.style.use('bmh')
plt.matplotlib.rc('text', usetex=True)
mpl.rcParams['figure.figsize'] = (14,8)

class my_star(my_stellar_core):

    def __init__(self,input_name=''):
        # Set parameters to default initial conditions
        self.set_to_default_initial_conditions(False)
        
        self.nabla_ad = 2/5              # Adiabatic temperature gradient ideal
        self.cp = 5*self.k_B/2/self.m_u/self.mu_0   # Specific heat cap const P

        # Control and program flow parameters
        self.given_initial_params = {'L0':self.L0,'M0':self.M0,'rho0':self.rho0,\
                                    'T0':self.T0,'R0':self.R0}
        self.read_opacity('../../project_1/python/opacity.txt')  # Calls read_opacity() in initialization
        self.name = input_name          # optional name for the object
        self.sanity = False             # boolean for use in sanity checks

    def ODE_solver(self,RHS=0,input_dm = -1e-20*Sun.M,variable_step=True):
        """
        A wrapper for solving the new system of ODE using the framework in p1
        @ RHS - A function returning the right-hand-side of the diff. system.
            Defaults to 0, in which case the self.get_RHS are used
        @ the other inputarguments are not used, sent to ODE_solver from p1
        
        Stores values for tracked quantities for each mass shell (done inside get_RHS)
        returns set of solutions of each parameter
        """
        if RHS==0:
            RHS = self.get_RHS
        # Create list to store tracked values at each mass shell
        self.Fc_list = []           # convective flux
        self.Fr_list = []           # radiative flux
        self.nabla_stable_list = [] # Nabla stable
        self.nabla_star_list = []   # Nabla star
        self.energy_PP1_list = []   # Energy from each reaction
        self.energy_PP2_list = []
        self.energy_PP3_list = []
        # Use same ODE_solver as in P1: (new RHS function!),
        # the convective stability check lies inside RHS which is call at each
        # mass shell!
        solutions = super().ODE_solver(RHS,input_dm,variable_step) 
        return solutions

    # ------------- Getters/setters ------------- #
    # ------------------------------------------- #
    def get_RHS(self,M,diff_params,eq_params):
        """
        Defines the right-hand-side of differential equations for project 2
        Checks for convenctive instability and changes the temperature equation if needed
        Also appends values to lists for quantities stored for each iteration
        """
        nabla_star = self.get_nabla_star()
        nabla_stable = self.get_nabla_stable()
        rhs = super().get_RHS(M,diff_params,eq_params)# Get the RHS from project 1

        convective = nabla_stable > self.nabla_ad   # Test for convective stability
        if convective:  # If convective change equation for temp gradient:
            rhs[2] = -nabla_star*self.T/self.get_Hp()*rhs[0]
        # Store the flux and nabla values for this r:
        self.append_tracked_quantities(nabla_stable,nabla_star)
        return rhs

    def get_nabla_stable(self):
        nominator = 3*self.L*self.get_kappa()*self.rho*self.get_Hp()
        denominator = 64*pi*self.R**2*self.sigma*self.T**4
        return nominator/denominator       

    def get_nabla_star(self):
        factor = 3*self.get_kappa()*self.rho*self.get_Hp()/(16*self.sigma*self.T**4)
        return factor*(self.L/(4*pi*self.R**2)-self.get_Fc())

    def get_nabla_parcel(self):
        return self.get_nabla_star()-self.get_xi()**2

    def get_xi(self):
        # roots = np.roots([1,U/lm**2,U**2*geometric/lm**3,\
        #                 U/lm**2*(self.nabla_ad-self.get_nabla_stable())])
        lm = self.get_lm()
        geometric = 4/lm    #S/Q/d = 2/r_p = 4/lm
        U = self.get_U()
        # Three roots, atleast one real:
        roots = np.roots([lm**2/U,1,U*geometric/lm,\
                        (self.nabla_ad-self.get_nabla_stable())])
        #Take out the real root, if there are more than one give error
        #This method is prefered over np.imag(roots)==0 as of numerical unstability
        real_roots = np.real(roots[np.abs(np.imag(roots))<1e-7])
        if len(real_roots)>1:
            print('------------------------------------------------')
            print('Found more than one real root in cubic equation!')
            print('------------------------------------------------')
            print(roots)
            print(real_roots)
            sys.exit()
        return float(real_roots)  # Return the real root

    def get_lm(self):
        return self.alpha*self.get_Hp()

    def get_v(self):
        return self.alpha/2*self.get_xi()*np.sqrt(self.get_Hp()*self.get_g())

    def get_g(self):
        return self.G*self.M/self.R**2

    def get_kappa(self):
        if self.sanity:  # Fixed value used in sanity checks
            kappa = 3.98    # [m^2 kg^-1]
        else:
            kappa = self.get_opacity(self.T,self.rho)
        return kappa

    def get_Fc(self):
        convective = self.get_nabla_stable() > self.nabla_ad   # Test for convective stability
        if not convective:  # Set the flux to zero if not convective
            return 0
        xi = self.get_xi()
        g = self.get_g()
        Hp = self.get_Hp()
        lm = self.get_lm()
        return self.rho*self.cp*self.T*np.sqrt(g)*Hp**(-3/2)*(lm/2)**2*xi**3

    def get_Fr(self):
        convective = self.get_nabla_stable() > self.nabla_ad   # Test for convective stability
        if not convective:  # If not convective, the radiation is the total flux
            return self.L/(4*pi*self.R**2)
        nominator = 16*self.sigma*self.T**4*self.get_nabla_star()
        denominator = 3*self.get_kappa()*self.rho*self.get_Hp()
        return nominator/denominator

    def get_U(self):
        nominator = 64*self.sigma*self.T**3*np.sqrt(self.get_Hp())
        denominator = 3*self.get_kappa()*self.rho**2*self.cp*np.sqrt(self.get_g())
        return nominator/denominator

    def get_Hp(self):
        return self.P/(self.rho*self.get_g())
        #return self.k_B*self.T/(self.get_g()*self.mu_0*self.m_u)

    def set_to_default_initial_conditions(self,set_param_selfs=False):
        """ Helper function to set all parameters to given intitial conditions """
        self.L0 = Sun.L             # [W]
        self.R0 = Sun.R             # [kg]
        self.M0 = Sun.M             # [m]
        self.rho0 = 1.42e-7*Sun.rho_avg # [kg m^-3]
        self.T0 = 5770              # [K]
        self.alpha = 1              # mix length parameter [0.5,2]
        # Mean molecular weight
        self.mu_0 = 1/(2*self.X+self.Y3+3/4*(self.Y-self.Y3)+1/2*self.Z)

        if set_param_selfs:
            self.P0 = self.get_P_EOS(self.rho0,self.T0)
            engine = stellar_engine()
            self.set_current_selfs([self.R0,self.L0,self.T0,self.P0,self.rho0,engine(self.rho0,self.T0),self.M0])

    def set_normalaized_energies(self):
        self.PP1_ar = np.asarray(self.energy_PP1_list)
        self.PP2_ar = np.asarray(self.energy_PP2_list)
        self.PP3_ar = np.asarray(self.energy_PP3_list)
        component_sum = self.PP1_ar+self.PP2_ar+self.PP3_ar
        self.PP1_ar /= component_sum
        self.PP2_ar /= component_sum
        self.PP3_ar /= component_sum

    # ------------ Convenient funcs ----------- #
    # ----------------------------------------- #
    def append_tracked_quantities(self,nabla_stable,nabla_star):
        """ Appends values at current shell for quantities tracked in project2 """
        self.Fc_list.append(self.get_Fc())
        self.Fr_list.append(self.get_Fr())
        self.nabla_stable_list.append(nabla_stable)
        self.nabla_star_list.append(nabla_star)
        self.energy_PP1_list.append(self.engine.energy_PP1)
        self.energy_PP2_list.append(self.engine.energy_PP2)
        self.energy_PP3_list.append(self.engine.energy_PP3)

    def print_selfs(self):
        print('M   = {:.3e}, M0     = {:.3e}'.format(self.M,self.M0))
        print('R   = {:.3e}, R0     = {:.3e}'.format(self.R,self.R0))
        print('L   = {:.3e}, L0     = {:.3e}'.format(self.L,self.L0))
        print('T   = {:.3e}, T0     = {:.3e}'.format(self.T,self.T0))
        print('P   = {:.3e}, P0     = {:.3e}'.format(self.P,self.P0))
        print('rho = {:.3e}, rho0   = {:.3e}'.format(self.rho,self.rho0))
        print('eps = {:.3e}'.format(self.epsilon))

    def save_solutions_to_file(self,solutions,filename):
        pass

    # ---------------- Plotters --------------- #
    # ----------------------------------------- #
    def cross_section(self,solutions,show_every=25):
        # --------------------------------------------------------------------
        # Assumptions:
        # --------------------------------------------------------------------
        # * R_values is an array of radii [unit: R_sun]
        # * L_values is an array of luminosities (at each r) [unit: L_sun]
        # * F_C_list is an array of convective flux ratios (at each r)
                #[unit: relative value between 0 and 1]
        # * n is the number of elements in both these arrays
        # * R0 is the initial radius
        # * (show_every = 50 worked well for me)
        # * core_limit = 0.995 (when L drops below this value, we are in the core)
        # --------------------------------------------------------------------
        r,L = solutions[:2]
        r = r/Sun.R
        L = L/Sun.L
        core_limit = 0.995
        F_C_list = self.Fc_list
        n = len(r)

        fig, ax = plt.subplots(1,1)

        rmax = 1.2
        ax.set_xlim(-rmax,rmax)
        ax.set_ylim(-rmax,rmax)
        ax.set_aspect('equal')  # make the plot circular
        j = show_every
        for k in range(0, n-1):
            j += 1
            if j >= show_every: # don't show every step - it slows things down
                if(L[k] > core_limit):   # outside core
                    if(F_C_list[k] > 0.0):      # convection
                        circR = plt.Circle((0,0),r[k],color='red',fill=False)
                        ax.add_artist(circR)
                    else:               # radiation
                        circY = plt.Circle((0,0),r[k],color='yellow',fill=False)
                        ax.add_artist(circY)
                else:               # inside core
                    if(F_C_list[k] > 0.0):      # convection
                        circB = plt.Circle((0,0),r[k],color='blue',fill = False)
                        ax.add_artist(circB)
                    else:               # radiation
                        circC = plt.Circle((0,0),r[k],color='cyan',fill = False)
                        ax.add_artist(circC)
                j = 0

        # only add one (the last) circle of each colour to legend:
        circR = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='red',fill=True)
        circY = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='yellow',fill=True)
        circC = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='cyan',fill=True)
        circB = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='blue',fill=True)
        fig.legend([circR, circY, circC, circB],['Convection outside core',\
            'Radiation outside core', 'Radiation inside core',r'$\nabla_s$' ])#'Convection inside core'])
        
        plt.title('Cross-section of star')
        #plt.show()

    # ------------- Sanity checks ------------- #
    # ----------------------------------------- #
    def example_sanity(self):
        """ Performs the sanity check from example 5.1 """

        # Set fixed values used in sanity check
        self.T0 = 0.9e6          # [K]
        self.rho0 = 55.9         # [kg m^-3]
        self.R0 = 0.84*Sun.R     # [m]
        self.M0 = 0.99*Sun.M     # [kg]
        self.L0 = Sun.L          # Luminosity
        self.alpha = 1          # Parameter
        self.mu_0 = 0.6         # Mean molecular weight
        self.sanity = True      # Set the sanity boolean to true

        self.P0 = self.get_P_EOS(self.rho0,self.T0)
        engine = stellar_engine()
        self.set_current_selfs([self.R0,self.L0,self.T0,self.P0,self.rho0,engine(self.rho0,self.T0),self.M0])

        goals = np.array([32.5e6,5.94e5,1.175e-3,0.400,65.62,0.88,0.12]) # Goal values
        # Computed values:
        Fc = self.get_Fc()
        Fr = self.get_Fr()
        FrFc = Fc+Fr
        calculated = np.array([self.get_Hp(),self.get_U(),self.get_xi(),\
                    self.get_nabla_star(),self.get_v(),Fc/FrFc,Fr/FrFc])
        
        self.comparing_table(calculated,goals)   # Print table

        print('{:^41s}'.format('Nablas'))
        print('Adiabatic |  Parcel  |   Star   |  Stable')
        print('{:^9.6f} < {:^6.6f} < {:^6.6f} < {:^6.6f}'.format(self.nabla_ad,\
            self.get_nabla_parcel(),self.get_nabla_star(),self.get_nabla_stable()))

        self.sanity = False     # Setting parameters back to default
        self.set_to_default_initial_conditions()

    def plot_sanity(self):
        """ Performs the plotted sanity check from app. E """
        self.set_to_default_initial_conditions(set_param_selfs=False)

        solutions = self.ODE_solver()
        self.set_normalaized_energies()
        #self.cross_section(solutions,15)
        r = solutions[0]
        Fc = np.asarray(self.Fc_list)
        Fr = np.asarray(self.Fr_list)
        FcFr = Fr+Fc

        legstr = [r'$\nabla_{\text{star}}$',r'$\nabla^{\star}$',r'$\nabla_{\text{ad}}$']
        fig, ax = plt.subplots(1,1)
        ax.semilogy(r/Sun.R,self.nabla_stable_list,color='tab:blue',\
                    label=legstr[0],alpha=0.6)
        ax.semilogy(r/Sun.R,self.nabla_star_list,color='tab:orange',\
                    label=legstr[1],alpha=0.6)
        ax.semilogy(r/Sun.R,np.ones_like(r)*self.nabla_ad,color='tab:green',\
                    label=legstr[2],alpha=0.6)
        ax.set_ylim(top=1e3)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles,labels)
        
        #fig.savefig('./../plots/plot_sanity.pdf')

        # plt.figure()
        # plt.plot(r,Fc/FcFr,label='Fc')
        # plt.plot(r,Fr/FcFr,label='Fr')
        # plt.legend()

        # plt.figure()
        # plt.title('Not currently working')
        # plt.plot(r,self.PP1_ar,label='PP1')
        # plt.plot(r,self.PP2_ar,label='PP2')
        # plt.plot(r,self.PP3_ar,label='PP3')
        # plt.legend()

if __name__ == '__main__':
    Star = my_star('Renate')
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'sanity' or sys.argv[1] == 'Sanity':  # Run sanity checks
            Star.example_sanity()
            Star.plot_sanity()
        if sys.argv[1] == 'experiment' or sys.argv[1] == 'Experiment':  # Run the initial experiments
            Star.experiment_multiple_solutions('T',low=0.2,high=5,nr=8)
    plt.show()
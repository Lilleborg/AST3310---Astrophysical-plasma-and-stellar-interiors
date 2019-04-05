from numpy import pi as pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp2d

import sys
sys.path.append("../../project_0")     # for own use when developing
sys.path.append("../../project_1")     # for own use when developing

from engine import stellar_engine,test_engine_1,test_engine_2
from star_core import my_stellar_core
import Solar_parameters as Sun      # Holding different given parameters for the Sun

# Plotting style
plt.style.use('bmh')
#plt.matplotlib.rc('text', usetex=True)
mpl.rcParams['figure.figsize'] = (14,8)

class my_star(my_stellar_core):

    def __init__(self,input_name=''):
        ## Some constants
        self.alpha = 1                   # parameter relating scale height and mix length

        # Mean molecular weight
        self.mu_0 = 1/(2*self.X+self.Y3+3/4*(self.Y-self.Y3)+1/2*self.Z)
        
        self.nabla_ad = 2/5              # Adiabatic temperature gradient ideal
        self.cp = 5*self.k_B/2/self.m_u/self.mu_0   # Specific heat cap const P

        # Set parameters to default initial conditions
        self.set_to_default_initial_conditions(True)

        # Control and program flow parameters
        self.given_initial_params = {'L0':self.L0,'M0':self.M0,'rho0':self.rho0,\
                                    'T0':self.T0,'R0':self.R0}
        self.read_opacity('../../project_1/opacity.txt')  # Calls read_opacity() in initialization
        self.name = input_name          # optional name for the object
        self.sanity = False             # boolean for use in sanity checks

    def ODE_solver2(self,RHS,input_dm = -1e20,variable_step=True):
        self.Fc_list = []   # Create list to store the convective flux values
        self.nabla_stable_list = []
        self.nabla_star_list = []
        solutions = super().ODE_solver(RHS,input_dm,variable_step)
        return solutions

    # ------------- Getters/setters ------------- #
    # ------------------------------------------- #
    def get_RHS_p2(self,M,diff_params,eq_params):
        """
        Defines the right-hand-side of differential equations for project 2
        Checks for convenctive instability and changes the temperature equation if
        """
        # print('RHS in p2')
        # print('m',M)
        # print('diff',diff_params)
        # print('eq',eq_params)
        convective = self.get_nabla_stable() > self.nabla_ad   # Test for convective stability
        rhs = self.get_RHS_p1(M,diff_params,eq_params)      # Get the RHS from project 1
        if convective:  # If convective change equation for temp gradient
            #print('Convective')
            rhs[2] = -self.get_nabla_star()*self.T/self.get_Hp()*rhs[0]
        else:
            #print('Not convective')
            #print(rhs)
            pass
        self.Fc_list.append(self.get_Fc())
        self.nabla_stable_list.append(self.get_nabla_stable())
        self.nabla_star_list.append(self.get_nabla_star())
        return rhs

    get_lm = lambda self: self.alpha*self.get_Hp()

    get_v = lambda self: self.alpha/2*np.sqrt(self.get_Hp()*self.get_g())*self.get_xi()

    get_g = lambda self: self.G*self.M/self.R**2

    def get_kappa(self):
        if self.sanity:  # Fixed value used in sanity checks
            kappa = 3.98    # [m^2 kg^-1]
        else:
            kappa = self.get_opacity(self.T,self.rho)
        return kappa

    def get_Fc(self):
        xi = self.get_xi()
        g = self.get_g()
        Hp = self.get_Hp()
        lm = self.get_lm()
        return self.rho*self.cp*self.T*np.sqrt(g)*Hp**(-3/2)*(lm/2)**2*xi**3

    def get_Fr(self):
        Hp = self.get_Hp()
        kappa = self.get_kappa()
        nominator = 16*self.sigma*self.T**4*self.get_nabla_star()
        denominator = 3*kappa*self.rho*self.get_Hp()
        return nominator/denominator

    def get_U(self):
        kappa = self.get_kappa()
        nominator = 64*self.sigma*self.T**3*np.sqrt(self.get_Hp())
        denominator = 3*kappa*self.rho**2*self.cp*np.sqrt(self.get_g())
        return nominator/denominator

    def get_Hp(self):
        return self.P/self.rho/self.get_g()

    def get_xi(self):
        lm = self.get_lm()
        geometric = 4/lm    #S/Q/d = 2/r_p = 4/lm
        U = self.get_U()
        # Coefficients:
        p = [1,U/lm**2,U**2*geometric/lm**2,U/lm**2*(self.nabla_ad-self.get_nabla_stable())]
        roots = np.roots(p)       # Three roots, atleast one real
        #Take out the real root, if there are more than one give error:
        real_roots = np.real(roots[np.abs(np.imag(roots))<1e-6])
        if len(real_roots)>1:
            print('------------------------------------------------')
            print('Found more than one real root in cubic equation!')
            print('------------------------------------------------')
            print(roots)
            print(real_roots)
            sys.exit()
        return float(real_roots)  # Return the real roots

    def get_nabla_stable(self):
        kappa = self.get_kappa()
        nominator = 3*self.L*kappa*self.rho*self.get_Hp()
        denominator = 64*pi*self.R**2*self.sigma*self.T**4
        return nominator/denominator

    def get_nabla_star(self):
        xi = self.get_xi()
        lm = self.get_lm()
        geometric = 4/lm    #S/Q/d
        U = self.get_U()
        return xi**2 + xi*U*geometric/lm + self.nabla_ad

    def get_nabla_parcel(self):
        return self.get_nabla_star()-self.get_xi()**2

    def set_to_default_initial_conditions(self,set_param_selfs=True):
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
            engine = stellar_engine(self.rho0,self.T0)
            self.set_current_selfs([self.R0,self.L0,self.T0,self.P0,self.rho0,engine(),self.M0])

    # ------------ Convenient funcs ----------- #
    # ----------------------------------------- #

    def print_selfs(self):
        print('M   = {:.3e}, M0     = {:.3e}'.format(self.M,self.M0))
        print('R   = {:.3e}, R0     = {:.3e}'.format(self.R,self.R0))
        print('L   = {:.3e}, L0     = {:.3e}'.format(self.L,self.L0))
        print('T   = {:.3e}, T0     = {:.3e}'.format(self.T,self.T0))
        print('P   = {:.3e}, P0     = {:.3e}'.format(self.P,self.P0))
        print('rho = {:.3e}, rho0   = {:.3e}'.format(self.rho,self.rho0))
        print('eps = {:.3e}'.format(self.eps))

    def save_solutions_to_file(self,solutions,filename):
        pass

    # ---------------- Plotters --------------- #
    # ----------------------------------------- #

    def cross_section(self,solutions,show_every=1):
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
        ax.legend([circR, circY, circC, circB],['Convection outside core',\
            'Radiation outside core', 'Radiation inside core', 'Convection inside core'])
        
        plt.title('Cross-section of star')
        #plt.show()

    # ------------- Sanity checks ------------- #
    # ----------------------------------------- #

    def example_sanity(self):
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
        engine = stellar_engine(self.rho0,self.T0)
        self.set_current_selfs([self.R0,self.L0,self.T0,self.P0,self.rho0,engine(),self.M0])

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

    def plot_sanity_p1(self):
        """ Performing the plot sanity check from app. D """

        test_object = my_star()#filename='../../project_1/opacity.txt')

        # Ensure same initial parameters as in app. D
        test_object.R0 = 0.72*Sun.R 
        test_object.rho0 = 5.1*Sun.rho_avg
        test_object.T0 = 5.7e6
        test_object.L0 = 1*Sun.L           # Luminosity [W]
        test_object.M0 = 0.8*Sun.M         # Mass [kg]

        # Set up axes and fig objects
        fig, ((Rax,Lax),(Tax,Dax)) = plt.subplots(2,2,figsize = (14.3,8),sharex='all')

        r,L,T,P,rho,eps,M = test_object.ODE_solver2(self.get_RHS_p2,variable_step=True)
        scaled_mass = M/Sun.M

        Rax.set_title('Radius vs mass')
        Rax.plot(scaled_mass,r/Sun.R)
        Rax.set_ylabel(r'$R/R_{sun}$')

        Lax.set_title('Luminosity vs mass')
        Lax.plot(scaled_mass,L/Sun.L)
        Lax.set_ylabel(r'$L/L_{sun}$')

        Tax.set_title('Temperature vs mass')
        Tax.plot(scaled_mass,T*1e-6)
        Tax.set_ylabel(r'$T[MK]$')
        Tax.set_xlabel(r'$M/M_{sun}$')

        Dax.set_title('Density vs mass')
        Dax.semilogy(scaled_mass,rho/Sun.rho_avg)
        Dax.set_ylabel(r'$\rho/\rho_{sun}$')
        Dax.set_xlabel(r'$M/M_{sun}$')
        Dax.set_ylim(1,10)

        fig.suptitle('Sanity check plot from app. D',size=30)
        fig.tight_layout(h_pad = 0.5,rect=[0, 0.03, 1, 0.95])
        #fig.savefig('./plots/plot_sanity.pdf')
        plt.show()

    def plot_sanity(self):
        self.set_to_default_initial_conditions(set_param_selfs=True)
    
        solutions = self.ODE_solver2(RHS=self.get_RHS_p2,variable_step=True,input_dm=-1e-20*Sun.M)
        self.cross_section(solutions)
        r = solutions[0]
        print(self.nabla_star_list)

        fig, ax = plt.subplots(1,1)
        ax.semilogy(r/Sun.R,self.nabla_star_list,color='tab:orange',\
                    label=r'$\nabla^\ast$')
        ax.semilogy(r/Sun.R,self.nabla_stable_list,color='tab:blue',\
                    label=r'$\nabla_{\text{stable}}$')
        ax.semilogy(r/Sun.R,np.ones_like(r)*self.nabla_ad,color='tab:green',\
                    label=r'$\nabla_{\text{ad}}$')
        ax.legend()
        plt.show()






if __name__ == '__main__':
    Star = my_star('Renate')
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'sanity' or sys.argv[1] == 'Sanity':  # Run sanity checks
            Star.example_sanity()
            #Star.opacity_sanity()
            #Star.plot_sanity_p1()
            #Star.plot_sanity()

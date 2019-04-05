from numpy import pi as pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

import sys
sys.path.append("../../project_0")     # for own use when developing
sys.path.append("../../project_1")     # for own use when developing

from engine import stellar_engine,test_engine_1,test_engine_2
from star_core import my_stellar_core
import Solar_parameters as Sun      # Holding different given parameters for the Sun

# Plotting style
plt.style.use('bmh')
plt.matplotlib.rc('text', usetex=True)

class my_Star(my_stellar_core):

    def __init__(self,input_name=''):
        ## Some constants
        self.m_u = 1.66053904e-27        # Unit atomic mass [kg]
        self.k_B = 1.382e-23             # Boltzmann's constant [m^2 kg s^-2 K^-1]
        self.sigma = 5.67e-8             # From app. A [W m^-2 K^-4]
        self.c = 2.998e8                 # speed of light [m s^-1]
        self.a = 4*self.sigma/self.c     # From app. A [J m^-3 K^-4]
        self.G = 6.672e-11               # Gravitational constant [N m^2 kg^-2]
        self.alpha = 1                   # parameter relating scale height and mix length

        # Mean molecular weight
        self.mu_0 = 1/(2*self.X+self.Y3+3/4*(self.Y-self.Y3)+1/2*self.Z)
        
        self.nabla_ad = 2/5              # Adiabatic temperature gradient ideal
        self.cp = 5*self.k_B/2/self.m_u/self.mu_0   # Specific heat cap const P

        # Set parameters to default initial conditions
        self.set_to_default_initial_conditions(has_mass_fractions=False)

        # Control and program flow parameters
        self.given_initial_params = {'L0':self.L0,'M0':self.M0,'rho0':self.rho0,'T0':self.T0,'R0':self.R0}
        super().read_opacity('../../project_1/opacity.txt')  # Calls read_opacity() in initialization
        self.name = input_name          # optional name for the object
        self.sanity = False             # boolean for use in sanity checks

    # ------------- Getters/setters ------------- #
    # ------------------------------------------- #
    get_lm = lambda self: self.alpha*self.get_Hp()

    get_v = lambda self: self.alpha/2*np.sqrt(self.get_Hp()*self.get_g())*self.get_xi()

    get_g = lambda self: self.G*self.M/self.r**2

    def get_kappa(self):
        if self.sanity:  # Fixed value used in sanity checks
            kappa = 3.98    # [m^2 kg^-1]
        else:
            kappa = super().get_opacity(self.T,self.rho)
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
        P = super().get_P_EOS(self.rho,self.T)
        return P/self.rho/self.get_g()

    def get_xi(self):
        lm = self.get_lm()
        geometric = 4/lm    #S/Q/d = 2/r_p = 4/lm
        U = self.get_U()
        # Coefficients:
        p = [1,U/lm**2,U**2*geometric/lm**2,U/lm**2*(self.nabla_ad-self.get_nabla_stable())]
        roots = np.roots(p)       # Three roots, atleast one real
        #Take out the real root, if there are more than one give error:
        real_roots = np.real(roots[np.abs(np.imag(roots))<1e-5])
        if len(real_roots)>1:
            print('------------------------------------------------')
            print('Found more than one real root in cubic equation!')
            print('------------------------------------------------')
            sys.exit()
        return float(real_roots)  # Return the real roots

    def get_nabla_stable(self):
        kappa = self.get_kappa()
        nominator = 3*self.L*kappa*self.rho*self.get_Hp()
        denominator = 64*pi*self.r**2*self.sigma*self.T**4
        return nominator/denominator

    def get_nabla_star(self):
        xi = self.get_xi()
        lm = self.get_lm()
        geometric = 4/lm    #S/Q/d
        U = self.get_U()
        return xi**2 + xi*U*geometric/lm + self.nabla_ad

    def get_nabla_parcel(self):
        return self.get_nabla_star()-self.get_xi()**2

    def set_to_default_initial_conditions(self,has_mass_fractions=True):
        """ Helper function to set all parameters to given intitial conditions """
        self.L0 = Sun.L             # [W]
        self.R0 = Sun.R             # [kg]
        self.M0 = Sun.M             # [m]
        self.rho0 = 1.42e-7*Sun.rho_avg # [kg m^-3]
        self.T0 = 5779              # [K]

        if not has_mass_fractions:
            # Mass fractions:
            self.X = 0.7                # H
            self.Y = 0.29               # He
            self.Y3 = 1e-10             # He3
            self.Z = 0.01               # Metals
            self.Z_Li = 1e-13           # Li7
            self.Z_Be = 1e-13           # Be7

    # ---------------- Plotters --------------- #
    # ----------------------------------------- #

    def cross_section(self,show_every=50):
        # -------------------------------------------------------------------------------------------------------
        # Assumptions:
        # -------------------------------------------------------------------------------------------------------
        # * R_values is an array of radii [unit: R_sun]
        # * L_values is an array of luminosities (at each r) [unit: L_sun]
        # * F_C_list is an array of convective flux ratios (at each r) [unit: relative value between 0 and 1]
        # * n is the number of elements in both these arrays
        # * R0 is the initial radius
        # * show_every is a variable that tells you to show every ...th step (show_every = 50 worked well for me)
        # * core_limit = 0.995 (when L drops below this value, we define that we are in the core)
        # -------------------------------------------------------------------------------------------------------
        fig, ax = plt.subplots(1,1)

        rmax = 1.2*self.R0
        ax.set_xlim(-rmax,rmax)
        ax.set_ylim(-rmax,rmax)
        ax.set_aspect('equal')  # make the plot circular
        j = show_every
        for k in range(0, n-1):
            j += 1
            if j >= show_every: # don't show every step - it slows things down
                if(L_values[k] > core_limit):   # outside core
                    if(F_C_list[k] > 0.0):      # convection
                        circR = plt.Circle((0,0),R_values[k],color='red',fill=False)
                        ax.add_artist(circR)
                    else:               # radiation
                        circY = plt.Circle((0,0),R_values[k],color='yellow',fill=False)
                        ax.add_artist(circY)
                else:               # inside core
                    if(F_C_list[k] > 0.0):      # convection
                        circB = plt.Circle((0,0),R_values[k],color='blue',fill = False)
                        ax.add_artist(circB)
                    else:               # radiation
                        circC = plt.Circle((0,0),R_values[k],color='cyan',fill = False)
                        ax.add_artist(circC)
                j = 0
        circR = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='red',fill=True)      # These are for the legend (drawn outside the main plot)
        circY = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='yellow',fill=True)
        circC = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='cyan',fill=True)
        circB = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='blue',fill=True)
        ax.legend([circR, circY, circC, circB], ['Convection outside core', 'Radiation outside core', 'Radiation inside core', 'Convection inside core']) # only add one (the last) circle of each colour to legend
        legend(loc=2)
        xlabel('')
        ylabel('')
        title('Cross-section of star')

        # Show all plots
        show()

    # ------------- Sanity checks ------------- #
    # ----------------------------------------- #

    
    def example_sanity(self):
        # Set fixed values used in sanity check
        self.T = 0.9e6          # [K]
        self.rho = 55.9         # [kg m^-3]
        self.r = 0.84*Sun.R     # [m]
        self.M = 0.99*Sun.M     # [kg]
        self.L = Sun.L          # Luminosity
        self.alpha = 1          # Parameter
        self.mu_0 = 0.6         # Mean molecular weight
        self.sanity = True      # Set the sanity boolean to true

        goals = np.array([32.5e6,5.94e5,1.175e-3,0.400,65.62,0.88,0.12]) # Goal values
        # Computed values:
        Fc = self.get_Fc()
        Fr = self.get_Fr()
        FrFc = Fc+Fr
        calculated = np.array([self.get_Hp(),self.get_U(),self.get_xi(),\
                    self.get_nabla_star(),self.get_v(),Fc/FrFc,Fr/FrFc])
        super().comparing_table(calculated,goals)   # Print table



        self.sanity = False     # Setting flow parameter back to default

    def plot_sanity(self):
        pass



if __name__ == '__main__':
    test = my_Star('Renate')
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'sanity' or sys.argv[1] == 'Sanity':  # Run sanity checks
            test.example_sanity()

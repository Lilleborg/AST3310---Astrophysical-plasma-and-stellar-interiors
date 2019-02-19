import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import sys
sys.path.append("../project_0")

from engine import stellar_engine,test_engine_1,test_engine_2
import Solar_parameters as Sun

# Plotting style
plt.style.use('bmh')
font = {'size'   : 16}
plt.matplotlib.rc('text', usetex=True)
plt.matplotlib.rc('font', **font)

class my_stellar_core():

    ## Some global constants
    m_u = 1.66053904e-27        # Unit atomic mass [kg]
    k_B = 1.382e-23             # Boltzmann's constant [m^2 kg s^-2 K^-1]
    sigma = 5.67e-8             # From app. A [W m^-2 K^-4]
    c = 2.998e8                 # speed of light [m s^-1]
    a = 4*sigma/c               # From app. A [J m^-3 K^-4]
    G = 6.672e-11               # Gravitational constant [N m^2 kg^-2]

    def __init__(self,debug=False):
        
        ### Initial parameters at bottom of solar convection zone
        ## Fixed parameters
        self.L0 = 1*Sun.L           # Luminosity [W]
        self.M0 = 0.8*Sun.M         # Mass [kg]
                                    # Mass fractions:
        self.X = 0.7                # H
        self.Y = 0.29               # He
        self.Y3 = 1e-10             # He3
        self.Z = 0.01               # Metals
        self.Z_Li = 1e-13           # Li7
        self.Z_Be = 1e-13           # Be7

        # Mean molecular weight
        self.mu_0 = 1/(2*self.X + self.Y3 + 3/4*(self.Y-self.Y3) + 4/7*self.Z_Li + 5/7*self.Z_Be)

        # Loose parameters
        self.rho0 = 5.1*Sun.rho_avg # Average density [kg m^-3]
        self.T0 = 5.7e6             # Temperature [T]
        self.R0 = 0.72*Sun.R        # Radius [m]

        ### Control and program flow parameters
        self.has_read_opacity_file = False
        self.debug = debug

    def read_opacity(self,filename='opacity.txt'):
        """
        Reads the file filename (default opacity.txt, dim 71x19) and interpolate the kappa values 
        Stores the cgs values log(R) [g cm^-3]  ?? Also [T^-3]??
        Log(T) in 1d arrays [K]
        Stores the cgs kappa values in a 2d array [cm^2 g^-1]
        Creates the function log_kappa_interp2d in cgs units to be used in get_opacity()
        """
        log_R = np.genfromtxt(filename,skip_footer=70)[1:]
        log_T = np.genfromtxt(filename,skip_header=2,usecols=0)
        log_kappa = np.genfromtxt(filename,skip_header=2, usecols=range(1,20))

        #print(log_R.shape,log_T.shape,log_kappa.shape)

        self.log_kappa_interp2d = interp2d(log_R,log_T,log_kappa,kind='linear')
        self.has_read_opacity_file = True

    def get_opacity(self,T,rho):
        """
        Uses the interpolated log_kappa function from read_opacity() and input
        @ T - temperate in [K]
        @ rho - density in [kg m^-3]
        calculates log_R from input args

        Returns the opacity values kappa in [m^2kg^-1]
        """
        if not self.has_read_opacity_file:
            self.read_opacity()

        rho_cgs = rho*1e-3
        R = rho_cgs/(T/1e6)**3

        log_R = np.log10(R)
        log_T = np.log10(T)

        log_kappa = float(self.log_kappa_interp2d(log_R,log_T)) # log10(kappa) in cgs
        kappa_SI = 10**log_kappa*1e-1                           # kappa in SI

        return kappa_SI
    
    def P_EOS(self,rho,T):
        """ Uses equation of state to find the pressure from density and temperature"""
        P_R = self.a/3*T**4      # Radiation pressure
        P_G = rho*self.k_B*T/self.mu_0/self.m_u     # Gass pressure from ideal gass law
        return P_R + P_G

    def rho_EOS(self,P,T):
        """ Uses equation fo state to find the density from pressure and temperature"""
        constant = self.mu_0*self.m_u/self.k_B  # Algebraic term

        return (P/T - self.a/3*T**3)*constant

    def ODE_solver(self,variable_step=False):


        ### List for storing integration values for each variable
        # ---------------- #
        r = [self.R0]
        P = [self.P_EOS(self.rho0,self.T0)]
        L = [self.L0]
        T = [self.T0]

        # Aslo need to update the density, energy production and mass
        rho = [self.rho0]
        M = [self.M0]
        engine = stellar_engine(rho[0],T[0])   # Energy production object (not the cleanest way of doing this...)
        eps = [engine()]
        # ---------------- #

        ### Integration loop using Euler
        



    # ------------- Sanity checks ------------- #
    def comparing_table(self,computed,expected):
        """ Helper function for comparing quantities computed vs expected in a table """
        print('{:^11s}|{:^11s}|{:^11s}'.format('Computed','Expected','Rel. error'))
        print('-'*35)
        try:
            for c,e in zip(computed,expected):
                rel_error = np.abs((e-c)/e)
                print('{:^11.3e}|{:^11.3e}|{:^11.6f}'.format(c,e,rel_error))

        except TypeError:   # If computed and expected is not iterable (e.i. floats)
            rel_error = np.abs((expected-computed)/expected)
            print('{:^11.3e}|{:^11.3e}|{:^11.6f}'.format(computed,expected,rel_error))
        print('-'*35)

    def opacity_sanity(self):
        """
        Executes the sanity check of the interpolation of kappa values. Checks using values from the table in appendix D for R and T if Kappa values match in both cgs and SI units.
        """
        self.read_opacity()
        # Values from the table
        log_R = np.array([6.,5.95,5.8,5.7,5.55,5.95,5.95,5.95,5.8,5.75,5.7,5.55,5.5])*-1
        log_T = np.array([3.75,3.755,3.755,3.755,3.755,3.77,3.78,3.795,3.77,3.775,3.78,3.795,3.8])

        # Values for testing get opacity
        T = 10**log_T
        R = 10**log_R

        rho_cgs = R*(T/1e6)**3
        rho_SI = rho_cgs*1e3

        log_k_cgs = np.array([1.55,1.51,1.57,1.61,1.67,1.33,1.2,1.02,1.39,1.35,1.31,1.16,1.11])*-1
        k_SI = np.array([2.84,3.11,2.68,2.46,2.12,4.7,6.25,9.45,4.05,4.43,4.94,6.89,7.69])*1e-3

        N = len(log_R)
        computed_cgs = np.zeros(N)
        computed_SI = np.zeros(N)
        computed_get_opacity = np.zeros(N)

        for i in range(N):
            computed_cgs[i] = self.log_kappa_interp2d(log_R[i],log_T[i])
            computed_get_opacity[i] = self.get_opacity(T[i],rho_SI[i])
        computed_SI = 10**computed_cgs*1e-1

        print('Directly from interpolation \nlog cgs:')
        self.comparing_table(computed_cgs,log_k_cgs)
        print('SI:')
        self.comparing_table(computed_SI,k_SI)
        print('\nThrough get_opacity() (SI-units)')
        self.comparing_table(computed_SI,k_SI)

    def EOS_sanity(self):
        """ Checks that the equation of state functions work both ways """
        expected_rho0 = self.rho0
        T0 = self.T0
        
        computed_P = self.P_EOS(expected_rho0,T0)
        computed_rho = self.rho_EOS(computed_P,T0)

        print('Density EOS')
        self.comparing_table(computed_rho,expected_rho0)

        expected_P0 = 5.2e14    # From app. A [Pa]

        computed_P = self.P_EOS(computed_rho,T0)

        print('Pressure EOS')
        self.comparing_table(computed_P,expected_P0)


if __name__ == '__main__':
    
    Star = my_stellar_core()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'sanity' or sys.argv[1] == 'Sanity':
            Star.opacity_sanity()
            Star.EOS_sanity()
    #Star.EOS_sanity()
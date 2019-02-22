from numpy import pi as pi
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

    # Mass fractions:
    X = 0.7                # H
    Y = 0.29               # He
    Y3 = 1e-10             # He3
    Z = 0.01               # Metals
    Z_Li = 1e-13           # Li7
    Z_Be = 1e-13           # Be7

    def __init__(self,debug=False,filename='opacity.txt'):
        
        ### Initial parameters at bottom of solar convection zone
        ## Fixed parameters
        self.L0 = 1*Sun.L           # Luminosity [W]
        self.M0 = 0.8*Sun.M         # Mass [kg]

        # Mean molecular weight
        self.mu_0 = 1/(2*self.X + self.Y3 + 3/4*(self.Y-self.Y3) + 4/7*self.Z_Li + 5/7*self.Z_Be)

        # Loose parameters
        self.rho0 = 5.1*Sun.rho_avg # Average density [kg m^-3]
        self.T0 = 5.7e6             # Temperature [T]
        self.R0 = 0.72*Sun.R        # Radius [m]
        #self.P0 = self.get_P_EOS(self.rho0,self.T0) # Pressure [Pa] found using equation of state

        ### Control and program flow parameters
        self.has_read_opacity_file = False
        self.debug = debug
        self.read_opacity(filename) # Calls read_opacity() in initialization

    def RHS_problem(self,M,diff_params,eq_params):
        r,L,T,P = diff_params
        rho,eps = eq_params
        #engine = stellar_engine(rho[-1],T[-1])
        return [1/(4*pi*r[-1]**2*rho[-1]),
                    eps[-1],
                    - 3*self.get_opacity(T[-1],rho[-1])*L[-1]/(256*pi**2*self.sigma*r[-1]**4*T[-1]**3),
                    - self.G*M[-1]/(4*pi*r[-1]**4)] # Order: dr, dL, dT, dP

    def ODE_solver(self,input_dm = -1e-4*Sun.M,variable_step=False,set_P0=0):
        """
        Solves the system of differential equations using Forward Euler
        UTVID***
        """
        engine = stellar_engine     # Create object for energy calculation
        ### List for storing integration values for each variable, starting with initial values
        r = [self.R0]
        L = [self.L0]
        T = [self.T0]
        P = [self.get_P_EOS(self.rho0,self.T0)]
        # if set_P0 != 0: # If P0 is not specified
        #     P = [set_P0]
        # else:
        #     P = [self.get_P_EOS(self.rho0,self.T0)]

        # print ('should be 0?',r[-1])
        # print ('shoudl be 2?',self.debug)
        # Aslo need to update the density, energy production and mass through the loop
        rho = [self.rho0]
        M = [self.M0]
        engine = stellar_engine(rho[0],T[0])   # Initialize engine with given rho and T
        epsilon = [engine()]                   # Calling engine gives the total energy

        diff_params = [r,L,T,P]                # List of parameters for diff. equations
        eq_params = [rho,epsilon]              # List of parameters found with own equations

        dm = input_dm       # Step mass size
        i = 0               # Keeping track of nr of iterations
        p = 1e-3            # Variable step size fraction
        broken = False      # Flow parameter, 
        print('Initial step size',dm,'in solar masses',dm/Sun.M)

        while M[-1] > 0 and M[-1]+dm>0:   # Integration loop using Euler until mass reaches zero
            # Finding right hand side values for diff eqs:
            d_params = np.asarray(self.RHS_problem(M,diff_params,eq_params))
            dr,dL,dT,dP = d_params  # Unpack for easier reading

            P.append(P[-1] + dm*dP)
            T.append(T[-1] + dm*dT)
            r.append(r[-1] + dm*dr)
            L.append(L[-1] + dm*dL)

            # Updates the rest of the parameters
            rho.append(self.rho_EOS(P[-1],T[-1]))
            engine = stellar_engine(rho[-1],T[-1])
            epsilon.append(engine())
            M.append(M[-1] + dm)

            if variable_step:
                # Getting an array with only the last elements of each parameter in diff_params
                current_last_values = np.asarray([item[-1] for item in diff_params])
                # Fractional change in variables
                dV_V = np.abs(current_last_values/d_params)        
                # If the change in any one variable is too high
                if np.any(dV_V>p):
                    dm = -np.min(np.abs(p*current_last_values/d_params))
            i += 1

            # Check for negative unphysical values
            current_last_values = np.asarray([item[-1] for item in diff_params+eq_params+[M]])
            if np.any(current_last_values<0):
                print(25*'#')
                print('Breaking at step {:d} with mass {:.3e} due to unphysical values'.format(i,M[-1]))
                print('R   = {:.3e}'.format(r[-1]))
                print('L   = {:.3e}'.format(L[-1]))
                print('T   = {:.3e}'.format(T[-1]))
                print('P   = {:.3e}'.format(P[-1]))
                print('rho = {:.3e}'.format(rho[-1]))
                print('eps = {:.3e}'.format(epsilon[-1]))
                print('Returning values up to not including last!')
                print(25*'#','\n')
                broken = True
                arrays = [] # Converting each list to arrays stored in the arrays-list
                for Q in diff_params + eq_params + [M]:
                    arrays.append(np.asarray(Q[:-1]))
                break

        print('Final step size  ',dm,'in solar masses',dm/Sun.M)
        
        if not broken:
            print('\n',25*'-')
            print('{:^25s}'.format('Final values of parameters'))
            print('{:>13s} {:.4e}'.format('Radius =',r[-1]))
            print('{:>13s} {:.4e}'.format('Luminosity =',L[-1]))
            print('{:>13s} {:.4e}'.format('Temperature =',T[-1]))
            print('{:>13s} {:.4e}'.format('Pressure =',P[-1]))
            print('{:>13s} {:.4e}'.format('Density =',rho[-1]))
            print('{:>13s} {:.4e}'.format('Epsilon =',epsilon[-1]))
            print('{:^25s}'.format('after {:d} iterations'.format(i)))
            print(25*'-')
            arrays = [] # Converting each list to arrays stored in the arrays-list
            for Q in diff_params + eq_params + [M]:
                arrays.append(np.asarray(Q))

        return arrays   # Order: r,L,T,P,rho,epsilon,M

    def experiment_multiple_solutions(self,varying_parameter,low=0.5,high=1.5,nr=6):
        """
        Solves the system multiple times while varying one parameter
        @ varying_parameter - string for witch parameter to change; one of [R,T,P,rho]
                                !!!Renamed to Q in the execution!!!
        """
        Q = varying_parameter+'0'       # Rename varying_parameter to Q
        attributes = self.__dict__      # dictionary of objects attributes, e.i. self.parameter
        original_value = attributes[Q]  # The original self.parameter value
        solutions = []                  # List for storing solutions
        new_initial_values = []
        #print(original_value)
        # Vary the given parameter from 0.2 to 1.5 of initial value in this loop
        for scale in np.linspace(low,high,nr):
            attributes[Q] = scale*original_value
            solutions.append(self.ODE_solver(variable_step=True))
        #print(original_value)
        attributes[Q] = original_value  # Set the self.parameter value back to original for fourther testing   
        self.plot_set_of_solutions(solutions,filename='test_change_'+Q+'.pdf',show=False,multible_parameters=Q)

    def plot_set_of_solutions(self,params,filename=0,show=False,multible_parameters=0):
        """
        Method for plotting a set of solutions for each parameter

        @ params - a list of lists with lists... here each element is a set of solutions,
                                                 where each set is a list of each parameter
                                                 makes it possible to plot multiple sets of solutions at once
        @ show - bool; show the figure or not at the end
        @ filename - bool; if not 0 the figure get save using filename in directory ./plots/
        """
        # Set up axes and fig objects
        fig, ((Pax,Lax,emptyax),(Rax,Tax,Dax)) = plt.subplots(2,3,figsize = (14,8),sharex='col')

        emptyax.remove()    # Don't display the [0,2] axes

        # Set up each axis object
        Rax.set_title('Radius vs mass')
        Rax.set_ylabel(r'$R/R_{sun}$')
        Rax.set_xlabel(r'$M/M_{sun}$')
        
        Lax.set_title('Luminosity vs mass')
        Lax.set_ylabel(r'$L/L_{sun}$')

        Tax.set_title('Temperature vs mass')
        Tax.set_ylabel(r'$T[MK]$')
        Tax.set_xlabel(r'$M/M_{sun}$')

        Dax.set_title('Density vs mass')
        Dax.set_ylabel(r'$\rho/\rho_{sun}$')
        Dax.set_xlabel(r'$M/M_{sun}$')
        
        Pax.set_title('Pressure vs mass')
        Pax.set_ylabel(r'$P [PPa]$')

        for set_of_solutions in params:
            r,L,T,P,rho,eps,M = set_of_solutions
            scaled_mass = M/Sun.M

            Rax.plot(scaled_mass,r/Sun.R,label=r'$R0 = {:.2f}R_\odot$'.format(r[0]/Sun.R))
            Lax.plot(scaled_mass,L/Sun.L)
            Tax.plot(scaled_mass,T*1e-6,label=r'$T0[MK] = {:.2f}$'.format(T[0]*1e-6))
            Dax.semilogy(scaled_mass,rho/Sun.rho_avg,label=r'$\rho0 = {:.2f}\rho_\odot$'.format(rho[0]/Sun.rho_avg))
            Pax.plot(scaled_mass,P*1e-15)

        if multible_parameters == 0:
            axes = [Rax,Tax,Dax]
            for ax in axes:
                ax.legend()
        else:
            multi = multible_parameters
            if multi == 'R0':
                handles, labels = Rax.get_legend_handles_labels()
                fig.suptitle('Experimenting with different initial radii',size=30)
            elif multi == 'T0':
                handles, labels = Tax.get_legend_handles_labels()
                fig.suptitle('Experimenting with different initial temperatures',size=30)
            elif multi == 'rho0':
                handles, labels = Dax.get_legend_handles_labels()
                fig.suptitle('Experimenting with different initial densities',size=30)
            else:
                print('Labels not understood!')
            fig.legend(handles, labels,loc='center',bbox_to_anchor=(4/5, 3/4))

        fig.tight_layout(rect=[0, 0.0, 1, 0.95],h_pad=0)
        if filename!=0:
            plt.savefig('./plots/'+filename)
        if show:
            plt.show()

    # --- Convenient functions and getters ---- #
    # ----------------------------------------- #
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
        rho_cgs = rho*1e-3
        R = rho_cgs/(T*1e-6)**3
        try:
            log_kappa = float(self.log_kappa_interp2d(np.log10(R),np.log10(T))) # log10(kappa) in cgs
            kappa_SI = 10**log_kappa*1e-1                           # kappa in SI
            return kappa_SI
    
        except:
            self.read_opacity()
            log_kappa = float(self.log_kappa_interp2d(np.log10(R),np.log10(T))) # log10(kappa) in cgs
            kappa_SI = 10**log_kappa*1e-1                           # kappa in SI
            return kappa_SI
    
    def get_P_EOS(self,rho,T):
        """ Uses equation of state to find the pressure from density and temperature"""
        P_R = self.a/3*T**4      # Radiation pressure
        P_G = rho*self.k_B*T/self.mu_0/self.m_u     # Gass pressure from ideal gass law
        return P_R + P_G

    def rho_EOS(self,P,T):
        """ Uses equation fo state to find the density from pressure and temperature"""
        constant = self.mu_0*self.m_u/self.k_B  # Algebraic term

        return (P/T - self.a/3*T**3)*constant

    # ------------- Sanity checks ------------- #
    # ----------------------------------------- #
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

    def plot_sanity(self):

        # Ensure same initial parameters as in app. D
        self.R0 = 0.72*Sun.R 
        self.rho0 = 5.1*Sun.rho_avg
        self.T0 = 5.7e6

        test_object = my_stellar_core()

        # Set up axes and fig objects
        fig, ((Rax,Lax),(Tax,Dax)) = plt.subplots(2,2,figsize = (14,8),sharex='all')

        r,L,T,P,rho,eps,M = test_object.ODE_solver(variable_step=True)
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
        fig.savefig('./plots/sanity_plot.pdf')
        plt.show()

    def opacity_sanity(self):
        """
        Executes the sanity check of the interpolation of kappa values. Checks using values from the table in appendix D for R and T if Kappa values match in both cgs and SI units.
        Also checks that the method get_opcaity() taking T and rho in SI units gives the same.
        """
        if not self.has_read_opacity_file:
            self.read_opacity()
        # Values from the table
        log_R = np.array([6.,5.95,5.8,5.7,5.55,5.95,5.95,5.95,5.8,5.75,5.7,5.55,5.5])*-1
        log_T = np.array([3.75,3.755,3.755,3.755,3.755,3.77,3.78,3.795,3.77,3.775,3.78,3.795,3.8])
        log_k_cgs = np.array([1.55,1.51,1.57,1.61,1.67,1.33,1.2,1.02,1.39,1.35,1.31,1.16,1.11])*-1
        k_SI = np.array([2.84,3.11,2.68,2.46,2.12,4.7,6.25,9.45,4.05,4.43,4.94,6.89,7.69])*1e-3

        # Values for testing get opacity
        T = 10**log_T
        R = 10**log_R

        rho_cgs = R*(T/1e6)**3
        rho_SI = rho_cgs*1e3

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
        self.comparing_table(computed_get_opacity,k_SI)

    def EOS_sanity(self):
        """ Checks that the equation of state functions work both ways """
        expected_rho0 = self.rho0
        T0 = self.T0
        
        computed_P = self.get_P_EOS(expected_rho0,T0)
        computed_rho = self.rho_EOS(computed_P,T0)

        print('Density EOS')
        self.comparing_table(computed_rho,expected_rho0)

        expected_P0 = 5.2e14    # From app. A [Pa]

        computed_P = self.get_P_EOS(computed_rho,T0)

        print('Pressure EOS')
        self.comparing_table(computed_P,expected_P0)


if __name__ == '__main__':
    
    Star = my_stellar_core()

    if len(sys.argv) > 1:
        if sys.argv[1] == 'sanity' or sys.argv[1] == 'Sanity':
            Star.opacity_sanity()
            Star.EOS_sanity()
            Star.plot_sanity()
            print('Sanity checks done, exiting')
            sys.exit(1)

    #Star.plot_set_of_solutions(Star.ODE_solver(variable_step=True),show=True)
    Star.experiment_multiple_solutions('T')
    #Star.experiment_multiple_solutions('R',low=0.3,high=0.9,nr=6)
    #Star.experiment_multiple_solutions('rho',low=0.2,high=5)
    
    plt.show()
    
    #Star.EOS_sanity()
from numpy import pi as pi
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker         # not functioning as intended
from scipy.interpolate import interp2d

import sys
sys.path.append("../project_0")     # for own use when developing

from engine import stellar_engine,test_engine_1,test_engine_2
import Solar_parameters as Sun      # Holding different given parameters for the Sun

# Plotting style
plt.style.use('bmh')
font = {'size'   : 16}
plt.matplotlib.rc('text', usetex=True)
plt.matplotlib.rc('font', **font)

class my_stellar_core(stellar_engine):

    ## Some global constants
    m_u = 1.66053904e-27        # Unit atomic mass [kg]
    k_B = 1.382e-23             # Boltzmann's constant [m^2 kg s^-2 K^-1]
    sigma = 5.67e-8             # From app. A [W m^-2 K^-4]
    c = 2.998e8                 # speed of light [m s^-1]
    a = 4*sigma/c               # From app. A [J m^-3 K^-4]
    G = 6.672e-11               # Gravitational constant [N m^2 kg^-2]

    # Mass fractions:
    X = 0.7                     # H
    Y = 0.29                    # He
    Y3 = 1e-10                  # He3
    Z = 0.01                    # Metals
    Z_Li = 1e-13                # Li7
    Z_Be = 1e-13                # Be7

    def __init__(self,filename='opacity.txt',input_name=''):
        # Mean molecular weight
        self.mu_0 = 1/(2*self.X+self.Y3+3/4*(self.Y-self.Y3)+1/2*self.Z)#+4/7*self.Z_Li+5/7*self.Z_Be+1/2*self.Z)
        
        # Initial parameters at bottom of solar convection zone
        # Fixed parameters
        self.L0 = 1*Sun.L           # Luminosity [W]
        self.M0 = 0.8*Sun.M         # Mass [kg]

        # Loose parameters
        self.rho0 = 5.1*Sun.rho_avg # Average density [kg m^-3]
        self.T0 = 5.7e6             # Temperature [T]
        self.R0 = 0.72*Sun.R        # Radius [m]
        
        # Control and program flow parameters
        self.given_initial_params = {'L0':self.L0,'M0':self.M0,'rho0':self.rho0,'T0':self.T0,'R0':self.R0}
        self.has_read_opacity_file = False
        self.read_opacity(filename) # Calls read_opacity() in initialization
        self.name = input_name      # optional name for the object

        # Found good initial parameters for radius, temperature and density
        # from find_my_star() method and manual inspection of plots
        self.R0074_T_rho = [[0.9,0.2],[0.74,0.2],[0.36,0.2],[0.2,0.40]]
        self.R0067_T_rho = [[0.9,0.2],[0.74,0.25],[0.59,0.2],[0.43,0.25],[0.36,0.3],[0.2,0.49]]
        self.R0059_T_rho = [[0.9,0.34],[0.74,0.34],[0.59,0.34],[0.43,0.34],[0.36,0.49],[0.28,0.7]]
        self.R0051_T_rho = [[0.9,0.55],[0.82,0.63],[0.67,0.63],[0.51,0.8],[0.43,1.],[0.36,1.21]]

    def ODE_solver(self,RHS,input_dm = -1e-4*Sun.M,variable_step=False):
        """
        Solves the system of differential equations using Forward Euler
        @ variable_step - True or False, wheter to use adaptive step or not
        @ input_dm - the initial value for step length
        """
        def print_progress():
            """ Quick function for printing progress """
            print('\nAt iteration {:d}'.format(iteration))
            print('dm  = {:.3e}'.format(dm))
            print('M   = {:.3e}, M/M0   = {:.3e}'.format(M[-1],M[-1]/M[0]))
            print('R   = {:.3e}, R/R0   = {:.3e}'.format(r[-1],r[-1]/r[0]))
            print('L   = {:.3e}, L/L0   = {:.3e}'.format(L[-1],L[-1]/L[0]))
            print('T   = {:.3e}, T/T0   = {:.3e}'.format(T[-1],T[-1]/T[0]))
            print('P   = {:.3e}, P/P0   = {:.3e}'.format(P[-1],P[-1]/P[0]))
            print('rho = {:.3e}, rho/rho0   = {:.3e}'.format(rho[-1],rho[-1]/rho[0]))
            print('eps = {:.3e}, eps/eps0   = {:.3e}\n'.format(epsilon[-1],epsilon[-1]/epsilon[0]))

        # Lists for storing integration values for each variable, starting with initial values:
        r = [self.R0]
        L = [self.L0]
        T = [self.T0]
        P = [self.get_P_EOS(self.rho0,self.T0)]# using equation of state to find Pressure
       
        # Aslo need lists for the density, energy production and mass through the loop:
        rho = [self.rho0]
        M = [self.M0]
        self.engine = stellar_engine()              # Create object for energy calculation
        epsilon = [self.engine(rho[0],T[0])]        # Calling engine gives the total energy
        diff_params = [r,L,T,P]                # List of parameters for diff. equations
        eq_params = [rho,epsilon]              # List of parameters found with own equations

        dm = input_dm       # Initial step mass size - changes if variable step length is active
        iteration = 0       # Keeping track of nr of iterations
        p = 0.01            # Variable step size fraction tolerance

        broken = False                         # Flow control parameter
        list_with_arrays = []                  # list holding the final solutions
        while M[-1]>0 and M[-1]+dm>0:   # Integration loop using Euler until mass reaches zero
            if (iteration%200 == 0 or iteration == 0):
                print_progress()
            # Update current self.parameter values:
            self.set_current_selfs([r[-1],L[-1],T[-1],P[-1],rho[-1],epsilon[-1],M[-1]]) 
            # Finding right hand side values for diff eqs:
            d_params = np.asarray(RHS(M,diff_params,eq_params)) # d_params is the f in the variable.pdf 
            if np.all(np.abs(d_params)<1e-30):  # If all the differentials are below this value break loop to save time
                break
            dr,dL,dT,dP = d_params  # Unpack for easier reading

            if variable_step:
                # To avoid overflow in division I set a minimum value on the d_params
                if np.any(np.abs(d_params)<1e-35):
                    # Still have to take into account the sign of the value:
                    for i,d_value in enumerate(d_params):
                        if np.abs(d_value)<1e-35:   
                            d_params[i] = d_value/np.abs(d_value) * 1e-35  # set the value to pluss/minus 1e-35

                # Getting an array with only the last elements of each parameter in diff_params
                current_last_values = np.asarray([item[-1] for item in diff_params])
                # Fractional change in variables
                dV_V = np.abs(current_last_values/d_params)        
                # If the fractional change in any one variable is higher than the tolerance adjust dm
                if np.any(dV_V>p):
                    dm = -np.min(np.abs(p*current_last_values/d_params))
                    if np.abs(dm)<1e12: # If the step size becomes too small break iterations
                        break           # This takes care of too asymptotal behavior and stops the loop

            # Update differential parameters:
            P.append(P[-1] + dm*dP)
            T.append(T[-1] + dm*dT)
            r.append(r[-1] + dm*dr)
            L.append(L[-1] + dm*dL)

            # Updates the rest of the parameters:
            rho.append(self.get_rho_EOS(P[-1],T[-1]))
            epsilon.append(self.engine(rho[-1],T[-1]))
            M.append(M[-1] + dm)

            # Check for negative unphysical values or if all the derivatives are approx zero:
            current_last_values = np.asarray([item[-1] for item in diff_params+eq_params+[M]])
            if np.any(current_last_values<0):
                print(25*'#','\nBreaking due to unphysical values')
                print_progress()
                print('Returning values up to not including last!\n',25*'#','\n')
                broken = True       
                for Q in diff_params + eq_params + [M]:
                    list_with_arrays.append(np.asarray(Q[:-1]))
                break

            iteration += 1
            if iteration==1:    # Debugging 
                print_progress()
            if iteration > 1e5:
                print('Breakig due to too many iterations')
                break
        
        if not broken:
            # If not broken, Calling RHS again, used in p2 RHS to get the last values:
            RHS(M,diff_params,eq_params)    
            print('\n',25*'-')
            print('{:^25s}'.format('Final values of parameters'))
            print_progress()
            print(25*'-')
            for Q in diff_params + eq_params + [M]:
                list_with_arrays.append(np.asarray(Q))

        return list_with_arrays   # Order: r,L,T,P,rho,epsilon,M

    def experiment_multiple_solutions(self,varying_parameter,low=0.5,high=1.5,nr=6,returning=False):
        """
        Solves the system multiple times while varying one parameter
        @ varying_parameter - string for witch parameter to change; one of [R,T,rho]
        @ returning - if the solutions should be returned or plotted directly
        @ low,high,nr - values used in linspace of the varying parameter
        """
        Q = varying_parameter+'0'       # Rename varying_parameter to Q
        attributes = self.__dict__      # dictionary of objects attributes, e.i. self.parameter
        solutions = []                  # List for storing solutions
        
        # Vary the given parameter of the original initial value in this loop
        for scale in np.linspace(low,high,nr):
            attributes[Q] = scale*self.given_initial_params[Q]
            solutions.append(self.ODE_solver(RHS=self.get_RHS, variable_step=True))
        
        attributes[Q] = self.given_initial_params[Q] # Set the self.parameter value back to original for fourther testing   
        if returning:
            return solutions
        else:
            self.plot_set_of_solutions(solutions,filename='plot_experiment_change_'+Q,multible_parameters=Q)

    def find_my_star(self,returning=False,tlow=0.2,thigh=0.5,tnr=10,rlow=0.2,rhigh=0.9,rnr=10,rholow=0.2,rhohigh=1.5,rhonr=10):
        """
        Looping over different temperatures and radii, using experiment_multiple_solutions() to vary rho and return the 
        solutions.
        Each set of solutions is passed through is_result_good() to evaluate if the goals of the simulations has been
        reached. If the goals are met, the initial parameters is saved to file for use later, and the hole set is plotted
        to see how the different rho-values behaives around the good temperature and radius value.

        @ returning - defaults to False, if True the good initial parameters are returned
        """

        def is_result_good(solution):
            """ quick function for testing if goals have been reached """
            r,L,T,P,rho,eps,M = solution
            values_approx_zero = False
            index_r0_goal_2 = np.argmin(np.abs(r-r[0]*0.1)) # index for where the radius is at 10%
            if r[-1]/r[0] <= 0.05 and L[-1]/L[0] <= 0.05 and M[-1]/M[0] <= 0.05 and L[index_r0_goal_2]<0.995*L[0]:
                values_approx_zero = True
            return values_approx_zero, r[0], T[0], rho[0]

        g_i_p = self.given_initial_params
        good_initial_parameters = []
        for scale_T in np.linspace(tlow,thigh,tnr):    # Varying initial temperature
            self.T0 = scale_T*g_i_p['T0']    
            for scale_R in np.linspace(rlow,rhigh,rnr):      # Varying initial radius
                self.R0= scale_R*g_i_p['R0']  

                string = '{:.2f}R0, {:.2f}T0'.format(scale_R,scale_T)   # String for file handling and plotting
                
                # Reusing writted method for changing rho parameter:
                solutions = self.experiment_multiple_solutions('rho',low=rholow,high=rhohigh,nr=rhonr,returning=True)
                
                solution_has_good_results = False   # Flow parameter, trigger plotting of a set of good solutions
                for one_solution in solutions:  # looping over the different sets of solutions stored in solutions
                    result_is_good,initial_r,initial_T,initial_rho = is_result_good(one_solution)    # Testing the solution
                    if result_is_good:  # If parameters are within 5% of initial value at the end and goal 2 is reached
                        good_initial_parameters.append([initial_r,initial_T,initial_rho])   # store the good initial parameter values
                        solution_has_good_results = True

                if solution_has_good_results and returning==False: # If solutions has a good solution and not used to return, plot it!
                    self.plot_set_of_solutions(solutions,multible_parameters='rho0',filename='plot_'+string.replace(', ','_'),title_string='using '+string)

                plt.close('all')    # Closing figures to not exhaust memory

        # Reset initial values
        self.T0 = g_i_p['T0']
        self.R0 = g_i_p['R0']
        self.rho0 = g_i_p['rho0']

        # Saving the sets of good parameters to file
        good_initial_parameters = np.asarray(good_initial_parameters)
        np.savetxt('good_initial_parameters.txt',good_initial_parameters,header='{:^24s}|{:^24s}|{:^24s}'.format('R0','T0','rho0')) 

        if returning:
            return good_initial_parameters

    def deeper_look_at_good_solutions(self,what_R0='0.59',final=0,Show=False,only_best=0):
        """
        Uses the good initial parameters found with find_my_star() to take a deeper look at the good solutions.
        @ what_R0 - string specifying what set of initial values to be used
        @ final - if 0 plot using the experimental plotter, if not 0 treat as the final model
        @ only_best - if not 0, only calculate the chosen best final solution
        """
        different_R0 = {'0.74':self.R0074_T_rho,'0.67':self.R0067_T_rho,'0.59':self.R0059_T_rho,'0.51':self.R0051_T_rho}
        g_i_p = self.given_initial_params
        
        if only_best != 0:
            what_R0 = 'best of 0.51'
            self.R0 = 0.51*g_i_p['R0']
            self.T0 = 0.82*g_i_p['T0']
            self.rho0 = 0.63*g_i_p['rho0']
            solutions = [self.ODE_solver(RHS=self.get_RHS,variable_step=True)]
        else:
            solutions = []  
            for T_rho in different_R0[what_R0]:
                self.R0 = float(what_R0)*g_i_p['R0']
                self.T0 = T_rho[0]*g_i_p['T0']
                self.rho0 = T_rho[1]*g_i_p['rho0']
                solutions.append(self.ODE_solver(RHS=self.get_RHS,variable_step=True))
        if final == 0:   # If not final, plot using experiment method
            self.plot_set_of_solutions(solutions,filename='plot_'+what_R0+'R0',multible_parameters='T0,rho0',title_string=values[i]+r'$R_0$',show=Show)
        else:
            self.plot_good_stars(solutions,filename='plot_good_star_new_mu'+what_R0+'R0',title_string=what_R0+r'$R_0$',show=Show)

        # Reset initial values
        self.T0 = g_i_p['T0']
        self.R0 = g_i_p['R0']
        self.rho0 = g_i_p['rho0']

    # --- Convenient functions, plotters and getters ---- #
    # --------------------------------------------------- #
    def get_RHS(self,M,diff_params,eq_params):
        """ Defines the right-hand-side of differential equations """
        # print('RHS in p1')
        # print('m',M)
        # print('diff',diff_params)
        # print('eq',eq_params)
        r,L,T,P = diff_params
        rho,eps = eq_params
        return [1/(4*pi*r[-1]**2*rho[-1]),
                    eps[-1],
                    - 3*self.get_opacity(T[-1],rho[-1])*L[-1]/(256*pi**2*self.sigma*r[-1]**4*T[-1]**3),
                    - self.G*M[-1]/(4*pi*r[-1]**4)] # Order: dr, dL, dT, dP

    def plot_set_of_solutions(self,solutions,filename=0,show=False,multible_parameters=0,title_string=''):
        """
        Method for plotting a set of solutions for each parameter

        @ solutions - a list of lists with lists... here each element is a set of solutions,
                                                 where each set is a list of each parameter
                                                 makes it possible to plot multiple sets of solutions at once
        @ show - bool; show the figure or not at the end
        @ filename - bool; if not 0 the figure get saved using filename in directory ./plots/
        @ multiible_parameters - used to determine legends when plotting multiple sets of solutions, if not 0
        @ title_string - add this string to the end of the supertitle used in each plot to keep track of specific parameters
        """
        # Set up axes and fig objects:
        fig, ((Pax,Lax,emptyax),(Rax,Tax,Dax)) = plt.subplots(2,3,figsize = (14,8),sharex='col')

        # Set up each axis object:
        emptyax.remove()    # Don't display the [0,2] axes

        Pax.set_title('Pressure vs mass')
        Pax.set_ylabel(r'$P [PPa]$')
        
        Lax.set_title('Luminosity vs mass')
        Lax.set_ylabel(r'$L/L_0$')

        Rax.set_title('Radius vs mass')
        Rax.set_ylabel(r'$R/R_0$')
        Rax.set_xlabel(r'$M/M_0$')

        Tax.set_title('Temperature vs mass')
        Tax.set_ylabel(r'$T[MK]$')
        Tax.set_xlabel(r'$M/M_0$')

        Dax.set_title('Density vs mass')
        Dax.set_ylabel(r'$\rho/\rho_0$')
        Dax.set_xlabel(r'$M/M_0$')

        g_i_p = self.given_initial_params   # Used for scaling in labels

        for set_of_solutions in solutions:
            r,L,T,P,rho,eps,M = set_of_solutions
            # Normalized/scaled quantyties:
            s_R,s_L,T,P,s_rho,s_eps,s_M = self.get_scale_parameter_lists(set_of_solutions)
            
            # Handling labels for different scenarios with the Rax axis:
            Rlabel = r'$R0 = {:.2f} R_0$'.format(r[0]/g_i_p['R0'])
            if multible_parameters == 'T0,rho0':
                Rlabel = r'$T0 = %.2f T_0, \rho 0 = %.2f \rho_0$'%(T[0]/g_i_p['T0'],rho[0]/g_i_p['rho0'])

            Rax.plot(s_M,s_R,label=Rlabel)
            Lax.plot(s_M,s_L)
            Tax.plot(s_M,T*1e-6,label=r'$T0 = {:.2f} T_0$'.format(T[0]/g_i_p['T0']))
            Dax.semilogy(s_M,s_rho,label=r'$\rho0 = {:.2f}\rho_0$'.format(rho[0]/g_i_p['rho0']))
            Pax.semilogy(s_M,P*1e-15)

        zero_goal_axis = [Lax,Rax]
        for ax in zero_goal_axis:     # Marking lines for goals for each parameter
            ax.axvline(x=0.05,color='r',linestyle='--',alpha=0.7)
            ax.axhline(y=0.05,color='r',linestyle='--',alpha=0.7)

        if multible_parameters in ['T0,rho0']:
            columnspacing = None
            nrcols = 1
            supertitle = 'Inspecting good solutions using '
            handles, labels = Rax.get_legend_handles_labels()

        if multible_parameters in ['R0','T0','rho0']:
            columnspacing = 0.5
            multi = multible_parameters
            supertitle = 'Experimenting with different initial '

            if multi == 'R0':
                handles, labels = Rax.get_legend_handles_labels()
                supertitle += 'radii '
            elif multi == 'T0':
                handles, labels = Tax.get_legend_handles_labels()
                supertitle += 'temperatures '
            elif multi == 'rho0':
                handles, labels = Dax.get_legend_handles_labels()
                supertitle += 'densities '
            else:
                print('Labels not understood!')
            nrcols = 1

        fig.legend(handles,labels,loc='lower left',bbox_to_anchor=((5)/7, 1/2),ncol=nrcols,columnspacing=columnspacing)
        fig.suptitle(supertitle + title_string, size=30)
        fig.tight_layout(rect=[0, 0, 1, 0.95],h_pad=0)

        if filename!=0:
            plt.savefig('./plots/'+filename+'.pdf')
        if show:
            plt.show()
   
    def plot_good_stars(self,solutions,filename=0,show=False,title_string=''):
        """
        Elaborate method for plotting the good final stars
        pretty much the same prosedure as plot_set_of_solutions and as these are basic python
        commands I will not comment further/again
        """

        fig, ((Pax,Lax,Eax),(Max,Tax,Dax)) = plt.subplots(2,3,figsize=(14,8),sharex='col')

        # Set up each axis object:
        Pax.set_title('Pressure vs radius')
        Pax.set_ylabel(r'$P [PPa]$')
        
        Lax.set_title('Luminosity vs radius')
        Lax.set_ylabel(r'$L/L_0$')

        Eax.set_title('Energy prod. vs radius')
        Eax.set_ylabel(r'$\varepsilon/\varepsilon_0$')

        Max.set_title('Mass vs radius')
        Max.set_ylabel(r'$M/M_0$')
        Max.set_xlabel(r'$R/R_0$')

        Tax.set_title('Temperature vs radius')
        Tax.set_ylabel(r'$T[MK]$')
        Tax.set_xlabel(r'$R/R_0$')

        Dax.set_title('Density vs radius')
        Dax.set_ylabel(r'$\rho/\rho_0$')
        Dax.set_xlabel(r'$R/R_0$')

        g_i_p = self.given_initial_params
        for set_of_solutions in solutions:
            r,L,T,P,rho,epsilon,M = set_of_solutions
            # Normalized/scaled quantyties:
            s_R,s_L,T,P,s_rho,s_eps,s_M = self.get_scale_parameter_lists(set_of_solutions)
            Label = r'$T0 = %.2f T_0$' '\n' r'$\rho 0 = %.2f \rho_0$'%(T[0]/g_i_p['T0'],rho[0]/g_i_p['rho0'])

            Pax.semilogy(s_R,P,label=Label)
            Lax.plot(s_R,s_L)
            Eax.semilogy(s_R,s_eps)
            Max.plot(s_R,s_M)
            Tax.plot(s_R,T*1e-6)
            Dax.semilogy(s_R,s_rho)

        index_core = np.argmin(np.abs(s_L-0.995))

        Lax.axhline(y=0.995,color='b',linestyle='--',alpha=0.7)
        for ax in [Pax,Lax,Eax,Max,Tax,Dax]:
            ax.axvline(x=s_R[index_core],color='b',linestyle='--',alpha=0.7)
            if ax == Lax or ax == Max:
                ax.axvline(x=0.05,color='r',linestyle='--',alpha=0.7)
                ax.axhline(y=0.05,color='r',linestyle='--',alpha=0.7)

        handles,labels = Pax.get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=len(solutions))
        fig.suptitle(r'Good star with $R0 = $ ' + title_string, size=30)
        fig.tight_layout(rect=[0, 0.06, 1, 0.97],h_pad=0)

        if filename != 0:
            plt.savefig('./plots/'+filename+'.pdf')
        if show:
            plt.show()

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

    def set_current_selfs(self,current_parameters):
        """
        Takes list of values and sets each self.parameter
        @ current_parameters - order: r,L,T,P,rho,epsilon,M
        """
        self.R,self.L,self.T,self.P,self.rho,self.epsilon,self.M = current_parameters

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

    def get_scale_parameter_lists(self,set_of_solutions):
        """ Helper function for scaling the parameters by their initial value """
        r,L,T,P,rho,epsilon,M = set_of_solutions
        return[r/r[0],L/L[0],T,P,rho/rho[0],epsilon/epsilon[0],M/M[0]]
    
    def get_P_EOS(self,rho,T):
        """ Uses equation of state to find the pressure from density and temperature """
        P_R = self.a/3*T**4      # Radiation pressure
        P_G = rho*self.k_B*T/self.mu_0/self.m_u     # Gass pressure from ideal gass law
        return P_R + P_G

    def get_rho_EOS(self,P,T):
        """ Uses equation fo state to find the density from pressure and temperature """
        constant = self.mu_0*self.m_u/self.k_B  # Algebraic term

        return (P/T - self.a/3*T**3)*constant

    # ------------- Sanity checks ------------- #
    # ----------------------------------------- #
    
    def plot_sanity(self):
        """ Performing the plot sanity check from app. D """
        print('heis')
        # Ensure same initial parameters as in app. D
        self.R0 = 0.72*Sun.R 
        self.rho0 = 5.1*Sun.rho_avg
        self.T0 = 5.7e6

        test_object = my_stellar_core()

        # Set up axes and fig objects
        fig, ((Rax,Lax),(Tax,Dax)) = plt.subplots(2,2,figsize = (14.3,8),sharex='all')

        r,L,T,P,rho,eps,M = test_object.ODE_solver(RHS=self.get_RHS,variable_step=True)
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
        fig.savefig('./plots/plot_sanity.pdf')
        plt.show()

    def opacity_sanity(self):
        """
        Executes the sanity check of the interpolation of kappa values.
        Checks using values from the table in appendix D for R and T if
        Kappa values match in both cgs and SI units.
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
        computed_rho = self.get_rho_EOS(computed_P,T0)

        print('Density EOS')
        self.comparing_table(computed_rho,expected_rho0)

        expected_P0 = 5.2e14    # From app. A [Pa]
        computed_P = self.get_P_EOS(computed_rho,T0)

        print('Pressure EOS')
        self.comparing_table(computed_P,expected_P0)


if __name__ == '__main__':
    
    Star = my_stellar_core(input_name='Renate') # As a tribute I name the star "Renate",
                                                # this is irrelevant for the calculations

    if len(sys.argv) > 1:
        if sys.argv[1] == 'sanity' or sys.argv[1] == 'Sanity':  # Run sanity checks
            Star.opacity_sanity()
            Star.EOS_sanity()
            Star.plot_sanity()

        if sys.argv[1] == 'experiment' or sys.argv[1] == 'Experiment':  # Run the initial experiments
            Star.experiment_multiple_solutions('T',low=0.2,high=5,nr=8)
            Star.experiment_multiple_solutions('R',low=0.2,high=1.5,nr=8)
            Star.experiment_multiple_solutions('rho',low=0.2,high=5,nr=8)
        
        if sys.argv[1] == 'findmystar' or sys.argv[1] == 'Findmystar':  # Run the method for finding all stable stars
            Star.find_my_star()

            # Test with higher values; no good solutions found using this command:
            #Star.find_my_star(thigh=5,tnr=4,rhigh=5,rnr=3,rhohigh=5,rhonr=5)

        if sys.argv[1] == 'deeperlook' or sys.argv[1] == 'Deeperlook':  # Run inspection of the good solutions
            Star.deeper_look_at_good_solutions()

        if sys.argv[1] == 'good' or sys.argv[1] == 'Good':  # Run final plotting of the best solutions
            Star.deeper_look_at_good_solutions(what_R0='0.51',final=1)
            Star.deeper_look_at_good_solutions(what_R0='0.59',final=1)
            Star.deeper_look_at_good_solutions(what_R0='0.67',final=1)
            Star.deeper_look_at_good_solutions(what_R0='0.74',final=1)
        
        if sys.argv[1] == 'onlygood' or sys.argv[1] == 'Onlygood':  # Run ONLY the chosen best solution
            Star.deeper_look_at_good_solutions(final=1,only_best=1,Show=True)
    
    
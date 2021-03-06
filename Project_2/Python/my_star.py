from numpy import pi as pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime

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
        self.project_2 = True
        self.break_goals = False
        self.datafiles_path = '../datafiles/'
        self.debug = False

        # Good initials from manual search:
        self.good_R0_scale = 1.1
        self.good_T0_scale = 0.95
        self.good_rho0_scale = 250

    def ODE_solver(self,RHS=0,input_dm = -1e-20*Sun.M,variable_step=True,goal_testing=False):
        """
        A wrapper for solving the new system of ODE using the framework in p1
        @ RHS - A function returning the right-hand-side of the diff. system.
            Defaults to 0, in which case the self.get_RHS are used
        @ the other inputarguments are not used, sent to ODE_solver from p1
        
        Stores values for tracked quantities for each mass shell (done inside get_RHS)
        returns set of solutions of each parameter
        """
        self.R_convective_end_goal = self.R0*0.85
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
        project2_values = [self.Fc_list,self.Fr_list,self.nabla_stable_list\
            ,self.nabla_star_list,self.energy_PP1_list,self.energy_PP2_list,self.energy_PP3_list]
        # Use same ODE_solver as in P1: (new RHS function!),
        # the convective stability check lies inside RHS which is call at each
        # mass shell!
        #print('for super ode',len(project2_values[2]),len(self.nabla_stable_list))
        solutions = super().ODE_solver(RHS,input_dm,variable_step,goal_testing=goal_testing) 
        #print('etter super ode',len(project2_values[2]),len(self.nabla_stable_list))
        return solutions,project2_values

    def experiment_deep_convection(self,use_goals=False,tlim=[0.6,5,6],rlim=[0.6,5,6],\
        rholim=[1,300,4],filename='plot_nabla_vary_',title='Experimenting with different intial '):
        """
        """
        g_i_p = self.given_initial_params
        hole_solutions = []
        for scale_R in np.linspace(*rlim):
            self.R0 = scale_R*g_i_p['R0']
            hole_solutions.append(super().experiment_multiple_solutions('rho',returning=True,\
                low=rholim[0],high=rholim[1],nr=rholim[2],usegoals=use_goals))
        self.plot_nablas_set_of_solutions(hole_solutions,exp_param='R',\
            title=title+'radii',filename=filename+'R')
        self.R0 = g_i_p['R0']

        hole_solutions = []
        for scale_T in np.linspace(*rlim):
            self.T0 = scale_T*g_i_p['T0']
            hole_solutions.append(super().experiment_multiple_solutions('rho',returning=True,\
                low=rholim[0],high=rholim[1],nr=rholim[2],usegoals=use_goals))
        self.plot_nablas_set_of_solutions(hole_solutions,exp_param='T',\
            title=title+'temperatures',filename=filename+'T')
        self.T0 = g_i_p['T0']

    def find_my_star(self,tlim=[0.8,1.1,5],rlim=[0.9,1.5,5],rholim=[5,300,20]):
        # low r for reaching core
        # low t for no convection in core
        # self.experiment_deep_convection(use_goals=True,tlim=tlim,rlim=rlim,rholim=rholim,\
        #     filename='good_star',title='test')
        good_initial_params_scales = []
        g_i_p = self.given_initial_params
        for scale_T in np.linspace(*tlim):
            self.T0 = scale_T*g_i_p['T0']
            for scale_R in np.linspace(*rlim):
                scale_R = 1.1
                self.R0 = scale_R*g_i_p['R0']

                for scale_rho in np.linspace(*rholim):
                    self.rho0 = scale_rho*g_i_p['rho0']
                    string = '{:.3f} {:.3f} {:.3f}\n'.format(scale_R,scale_T,scale_rho)
                    print(string)
                    sol = self.ODE_solver(goal_testing=True)
                    if not np.all(np.isnan(sol[0])):
                        good_initial_params_scales.append(string)
        #if len(good_initial_params_scales)>0:
        with open('./../datafiles/good_initial_scales.txt','a') as ofile:
            for i,str_ in enumerate(good_initial_params_scales):
                ofile.write(str_)
            print('./../datafiles/good_initial_scales.txt saved.')

    def view_good_experiment(self):
        g_i_p = self.given_initial_params
        filename = './../datafiles/good_initial_scales.txt'
        scale_R = np.genfromtxt(filename,skip_header=1,usecols=0)
        scale_T = np.genfromtxt(filename,skip_header=1,usecols=1)
        scale_rho = np.genfromtxt(filename,skip_header=1,usecols=2)
        
        for R,T,RHO in zip(scale_R,scale_T,scale_rho):
            #if R == 1.233 and T == 1.000:  # From trying different sets
            #if R == 1.167 and T == 1.000:
            if R == 1.100 and T == 0.875:
                self.R0 = R*g_i_p['R0']
                self.T0 = T*g_i_p['T0']
                self.rho0 = RHO*g_i_p['rho0']
                string = '{:.3f}R0 {:.3f}T0 {:.3f}rho0'.format(R,T,RHO)
                sol = self.ODE_solver()

                R = sol[0][0]
                legstr = [r'$\nabla_{\text{stable}}$',r'$\nabla^{\star}$',r'$\nabla_{\text{ad}}$']
                fig, ax = plt.subplots(1,1)
                ax.semilogy(R/self.R0,self.nabla_stable_list,label=legstr[0])
                ax.semilogy([0,R[0]/self.R0],[self.nabla_ad,self.nabla_ad],label=legstr[2])
                fig.suptitle('Inspecting '+string)
                fig.savefig('./../plots/plot_nabla_'+string.replace(' ','_')+'.pdf')
                plt.close('all')

    def final_star(self):
        """
        I did not have time to write this method more elegant by reusing existing methods.
        Uses the good initial parameters from manual search, solves the system and plots
        the required figures.
        """
        self.set_to_default_initial_conditions(False)
        g_i_p = self.given_initial_params
        self.R0 = self.good_R0_scale*g_i_p['R0']
        self.T0 = self.good_T0_scale*g_i_p['T0']
        self.rho0 = self.good_rho0_scale*g_i_p['rho0']
        string = '{:.3f}R0 {:.3f}T0 {:.3f}rho0'.format(self.good_R0_scale,self.good_T0_scale,self.good_rho0_scale)
        solution = self.ODE_solver()
        R,L,T,P,rho,epsilon,M = solution[0]
        s_R,s_L,T,P,s_rho,s_eps,s_M = self.get_scaled_parameter_lists(solution[0])
        Fc = np.asarray(self.Fc_list)
        Fr = np.asarray(self.Fr_list)
        FcFr = Fr+Fc
        convection_end = np.argmin(Fc) # Get the first index where Fc=0
        index_core = np.argmin(np.abs(s_L-0.995))   # Get index of core
        print('Chosen best model')
        print('Convection into {:.3f}'.format(s_R[convection_end]))
        print('Core reaching {:.3f}'.format(s_R[index_core]))

        ######
        # Plotting main parameters and cross section
        fig, ((Pax,Lax,crossax),(Max,Tax,Dax)) = plt.subplots(2,3)
        # Set up each axis object:
        Pax.set_title('Pressure vs radius')
        Pax.set_ylabel(r'$P [PPa]$')
        
        Lax.set_title('Luminosity vs radius')
        Lax.set_ylabel(r'$L/L_0$')

        crossax.set_title('Cross section')
        rmax = 1.1*s_R[0]
        crossax.set_xlim(-rmax,rmax)
        crossax.set_ylim(-rmax,rmax)
        crossax.set_aspect('equal')

        Max.set_title('Mass vs radius')
        Max.set_ylabel(r'$M/M_0$')
        Max.set_xlabel(r'$R/R_0$')

        Tax.set_title('Temperature vs radius')
        Tax.set_ylabel(r'$T[MK]$')
        Tax.set_xlabel(r'$R/R_0$')

        Dax.set_title('Density vs radius')
        Dax.set_ylabel(r'$\rho/\rho_0$')
        Dax.set_xlabel(r'$R/R_0$')

        # Filling in the cross section
        show_every = 15
        j = show_every
        for k in range(0,len(R)-1):
            j += 1
            if j >= show_every: # don't show every step - it slows things down
                if(s_L[k] > 0.995):   # outside core
                    if(Fc[k] > 0.0):      # convection
                        circR = plt.Circle((0,0),s_R[k],color='red',fill=False)
                        crossax.add_artist(circR)
                    else:               # radiation
                        circY = plt.Circle((0,0),s_R[k],color='yellow',fill=False)
                        crossax.add_artist(circY)
                else:               # inside core
                    if(Fc[k] > 0.0):      # convection
                        circB = plt.Circle((0,0),s_R[k],color='blue',fill = False)
                        crossax.add_artist(circB)
                    else:               # radiation
                        circC = plt.Circle((0,0),s_R[k],color='cyan',fill = False)
                        crossax.add_artist(circC)
                j = 0

        Pax.semilogy(s_R,P)
        Lax.plot(s_R,s_L)
        Max.plot(s_R,s_M)
        Tax.plot(s_R,T*1e-6)
        Dax.semilogy(s_R,s_rho)

        for ax in [Pax,Lax,Max,Tax,Dax]:
            ax.axvline(x=s_R[index_core],color='b',linestyle='--',alpha=0.5)
            ax.axvline(x=s_R[convection_end],color='r',linestyle='--',alpha=0.5)

        fig.suptitle('Main parameters and cross section, '+r'$R0 = %sR_0, T0 = %sT_0, \rho 0 = %s\rho_0$'\
            %(str(self.good_R0_scale),str(self.good_T0_scale),str(self.good_rho0_scale)))
        fig.savefig('./../plots/best/plot_params_cross_best'+string.replace(' ','_')+'.pdf')
        print('./../plots/best/plot_params_cross_best'+string.replace(' ','_')+'.pdf saved.')
        
        ####
        # Plotting flux fractions and temperature gradients
        legstr = [r'$\nabla_{\text{stable}}$',r'$\nabla^{\star}$',r'$\nabla_{\text{ad}}$']
        fig, ax = plt.subplots(3,1,figsize=(10,8))

        ax[0].plot(s_R,Fc/FcFr,label=r'$\frac{Fc}{Fc+Fr}$')
        ax[0].plot(s_R,Fr/FcFr,label=r'$\frac{Fr}{Fc+Fr}$')
        ax[0].legend()
        ax[0].set_ylabel('Fractions of flux')

        for i in range(0,3):
            ax[i].axvline(x=s_R[index_core],color='b',linestyle='--',alpha=0.5)
            ax[i].axvline(x=s_R[convection_end],color='r',linestyle='--',alpha=0.5)
            if i > 0:
                ax[i].semilogy(s_R,self.nabla_stable_list,label=legstr[0])
                ax[i].semilogy(s_R,self.nabla_star_list,label=legstr[1])
                ax[i].semilogy([0,1],[self.nabla_ad,self.nabla_ad],'-.',label=legstr[2],color='k')
                ax[i].legend()
                ax[i].set_xlabel(r'$R/R_0$')
                ax[i].set_ylabel(r'$\nabla$')
        ax[1].set_ylim(1e-2,np.max(self.nabla_stable_list)*2)
        ax[2].set_ylim(self.nabla_star_list[convection_end]*0.98,np.max(self.nabla_star_list)*1.02)
        ax[2].set_xlim(s_R[convection_end]*0.98,s_R[0]*1.02)
        ax[2].set_title('Magnifying the convection zone')

        fig.suptitle('Temperature gradients and fractions of heat transportation')
        fig.savefig('./../plots/best/plot_nablas_best'+string.replace(' ','_')+'.pdf')
        print('./../plots/best/plot_nablas_best'+string.replace(' ','_')+'.pdf')
        
        ######
        # Plotting energy chains
        self.set_normalaized_energies()        
        fig, ax = plt.subplots(3,1,figsize=(10,8))
        for i in [0,2]:
            ax[i].plot(s_R,epsilon/np.max(epsilon),label=r'$\varepsilon/\varepsilon_{\text{max}}$')
            ax[i].plot(s_R,self.PP1_ar,label='PP-1')
            ax[i].plot(s_R,self.PP2_ar,label='PP-2')
            ax[i].plot(s_R,self.PP3_ar,label='PP-3')
            ax[i].set_ylabel(r'$\varepsilon$ scaled')
        ax[0].legend()
        ax[2].set_title('Magnifying the core')
        ax[2].set_xlim(-0.009,s_R[index_core]*1.02)
        ax[2].set_xlabel(r'$R/R_0$')

        ax[1].plot(s_R,T*1e-6)
        ax[1].set_ylabel(r'$T[MK]$')
        ax[1].set_xlabel(r'$R/R_0$')
        ax[1].set_title('Temperature vs mass')
        for i in range(0,3):
            ax[i].axvline(x=s_R[index_core],color='b',linestyle='--',alpha=0.5)
            ax[i].axvline(x=s_R[convection_end],color='r',linestyle='--',alpha=0.5)
        fig.suptitle('Relative energy production from each PP-chain')
        fig.savefig('./../plots/best/plot_energy_best'+string.replace(' ','_')+'.pdf')
        print('./../plots/best/plot_energy_best'+string.replace(' ','_')+'.pdf')
        plt.show()

    # ---------------- Plotters --------------- #
    # ----------------------------------------- #
    def plot_nablas_set_of_solutions(self,set_of_solutions,title=0,exp_param=0,filename=0):
        g_i_p = self.given_initial_params   # Used for scaling in labels
        mpl.rcParams['figure.constrained_layout.use']= False
        fig, ax = plt.subplots(2,3,sharex='col',sharey='row',squeeze=False)
        i = 0
        n = 0    # number of solutions
        #super_add = ''   # Dummy variable
        for j,different_sets in enumerate(set_of_solutions):   
            for k,solution in enumerate(different_sets):
                n += 1
                if j > 2:
                    i = 1
                    j -= 3
                R,_,T,_,rho,_,M = solution[0]
                s_R,_,T,_,s_rho,_,s_M = self.get_scaled_parameter_lists(solution[0])
                nabla_stb = solution[1][2]
                ax[i,j].semilogy(s_R,nabla_stb,label=r'$\rho0 = {:.2f}\rho_0$'.format(rho[0]/g_i_p['rho0']))
                
                if exp_param == 'R':
                    ax[i,j].set_title(r'$R0 = %.2f R_0$'%(R[0]/g_i_p['R0']))
                    #super_add = 'radii'
                if exp_param == 'T':
                    ax[i,j].set_title(r'$T0 = %.2f T_0$'%(T[0]/g_i_p['T0']))
                    #super_add = 'temperatures'

                if k == 0:
                    ax[i,j].semilogy([0,1],[self.nabla_ad,self.nabla_ad],'-.',color='k')
                    ax[i,j].set_ylim(0.1,1e5)
                    for A in range(3):
                        ax[1,A].set_xlabel(r'$R/R_0$')
                        if A < 2:
                            ax[A,0].set_ylabel(r'$\nabla_{\text{stable}}$')
        if n != 0:
            handles,labels = ax[i,j].get_legend_handles_labels()    
            fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=n)
            fig.suptitle(str(title))
            fig.tight_layout(rect=[0, 0.06, 1, 0.95],h_pad=1)
            if filename != 0:
                fig.savefig('./../plots/'+filename + '.pdf')
                print('./../plots/'+filename + '.pdf saved.')

    def plot_cross_section(self,solutions,show_every=25,title=0,filename=0,scale_with_sun=False,show=False):
        """
        Modified version of cross_section.py
        # --------------------------------------------------------------------
        # * F_C_list is an array of convective flux ratios (at each r)
                #[unit: relative value between 0 and 1]
        # * n is the number of elements
        # * (show_every = 50 worked well for me)
        # * core_limit = 0.995 (when L drops below this value, we are in the core)
        # --------------------------------------------------------------------
        """
        if self.break_goals:    # If run is broken dont try to plot it!
            return
        g_i_p = self.given_initial_params   # Used for scaling in labels
        r,L = solutions[0][:2]
        if scale_with_sun:
            r = r/Sun.R
            L = L/Sun.L
            lab = r'$R/R_\odot$'
        else:
            r = r/g_i_p['R0']
            L = L/g_i_p['L0']
            lab = r'$R/R_0$'
        core_limit = 0.995
        F_C_list = solutions[1][0] #self.Fc_list
        n = len(r)

        fig, ax = plt.subplots(1,1)
        ax.set_ylabel(lab)
        ax.set_xlabel(lab)
        rmax = 1.2*r[0]
        ax.set_xlim(-rmax,rmax)
        ax.set_ylim(-rmax,rmax)
        ax.set_aspect('equal')  # make the plot circular
        
        counter = np.zeros(4)   # Tracks number of times in each region
        j = show_every
        for k in range(0, n-1):
            j += 1
            if j >= show_every: # don't show every step - it slows things down
                if(L[k] > core_limit):   # outside core
                    if(F_C_list[k] > 0.0):      # convection
                        circR = plt.Circle((0,0),r[k],color='red',fill=False)
                        ax.add_artist(circR)
                        counter[0] += 1
                    else:               # radiation
                        circY = plt.Circle((0,0),r[k],color='yellow',fill=False)
                        ax.add_artist(circY)
                        counter[1] += 1
                else:               # inside core
                    if(F_C_list[k] > 0.0):      # convection
                        circB = plt.Circle((0,0),r[k],color='blue',fill = False)
                        ax.add_artist(circB)
                        counter[2] += 1
                    else:               # radiation
                        circC = plt.Circle((0,0),r[k],color='cyan',fill = False)
                        ax.add_artist(circC)
                        counter[3] += 1
                j = 0
            # if k == n-2:
            #     circempty = plt.Circle((0,0),r[k],color='white',fill=True)
            #     ax.add_artist(circempty)

        circR = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='red',fill=True)      # These are for the legend (drawn outside the main plot)
        circY = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='yellow',fill=True)
        circC = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='cyan',fill=True)
        circB = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='blue',fill=True)
        fig.legend([circR, circY, circC, circB],['Convection outside core',\
            'Radiation outside core', 'Radiation inside core','Convection inside core'])
    
        if title != 0:
            plt.suptitle(title)
        else:
            plt.suptitle('Cross-section of star')
        if filename != 0:
            print('./../plots/'+filename + '.pdf saved.')
            fig.savefig('./../plots/'+filename + '.pdf')
        if True:#self.debug:  # Printing number of time in each region for debugging
            print('Hits in')
            print('Convection outside',counter[0])
            print('Radiation  outside',counter[1])
            print('Convection  inside',counter[2])
            print('Radiation   inside',counter[3])
            #plt.show()

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
        self.mu_0 = 1/(2*self.X+self.Y3+3/4*(self.Y-self.Y3)+1/2*(self.Z-self.Z_Li-self.Z_Be)\
                    +4/7*self.Z_Li+5/7*self.Z_Be)
        self.given_initial_params = {'L0':self.L0,'M0':self.M0,'rho0':self.rho0,'T0':self.T0,'R0':self.R0}
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
    def is_result_good(self,solution,inODE=False,final=False):
        if not final:   # Testing only for convection zone
            if self.R > self.R_convective_end_goal and self.R < 0.99*self.R0:
                is_convective = self.Fc_list[-1]> 0
                if not is_convective:   # If 
                    print(40*'-')
                    print('Breaking')
                    print('Initial convective zone not deep enough')
                    self.break_goals = True
        else:
            values_approx_zero=super().is_result_good(solution,final=True)[0]
            if not values_approx_zero:
                print(41*'-')
                print('Breaking iterations as goals near core are vialated')
                print(41*'-')
                self.break_goals = True

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
        sol = solutions[0]+solutions[1]
        np.savez_compressed(self.datafiles_path+filename,*sol)

    def load_solutions_from_file(self,filename):
        solutions = np.load(self.datafiles_path+filename)
        R = solutions['arr_0']
        L = solutions['arr_1']
        T = solutions['arr_2']
        P = solutions['arr_3']
        rho = solutions['arr_4']
        eps = solutions['arr_5']
        M = solutions['arr_6']
        Fc = solutions['arr_7']
        Fr = solutions['arr_8']
        nabla_stable = solutions['arr_9']
        return [R,L,T,P,rho,eps,M],[Fc]

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
        #self.mu_0 = 0.618
        solutions = self.ODE_solver(goal_testing=False)
        self.set_normalaized_energies()

        self.plot_cross_section(solutions,1,title='Cross section sanity test',filename='plot_sanity_cross_section')
        R = solutions[0][0]
        Fc = np.asarray(self.Fc_list)
        Fr = np.asarray(self.Fr_list)
        FcFr = Fr+Fc

        legstr = [r'$\nabla_{\text{stable}}$',r'$\nabla^{\star}$',r'$\nabla_{\text{ad}}$']
        fig, ax = plt.subplots(1,1)
        ax.semilogy(R/Sun.R,self.nabla_stable_list,color='tab:blue',\
                    label=legstr[0],alpha=0.6)
        ax.semilogy(R/Sun.R,self.nabla_star_list,color='tab:orange',\
                    label=legstr[1],alpha=0.6)
        ax.semilogy(R/Sun.R,np.ones_like(R)*self.nabla_ad,color='tab:green',\
                    label=legstr[2],alpha=0.6)
        ax.set_ylim(top=1e3)
        ax.set_xlabel(r'$R/R_0 = R/R_\odot$')
        ax.set_ylabel(r'$\nabla$')
        ax.legend()
        fig.suptitle('Temperature gradients sanity check')
        
        fig.savefig('./../plots/plot_sanity.pdf')
        print('./../plots/plot_sanity.pdf saved:')

        #handles, labels = ax.get_legend_handles_labels()
        #fig.legend(*ax.get_legend_handles_labels())#handles,labels)
        

if __name__ == '__main__':
    Star = my_star('Renate')
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'sanity' or sys.argv[1] == 'Sanity':  # Run sanity checks
            Star.example_sanity()
            Star.plot_sanity()
        if sys.argv[1] == 'experiment' or sys.argv[1] == 'Experiment':  # Run the initial experiments
            Star.experiment_deep_convection(use_goals=False)
        if sys.argv[1] == 'findstar' or sys.argv[1] == 'Findstar':
            Star.find_my_star()
        if sys.argv[1] == 'viewgood' or sys.argv[1] == 'Viewgood':
            Star.view_good_experiment()
        if sys.argv[1] == 'final' or sys.argv[1] == 'final':
            Star.final_star()

    plt.show()

"""
    
"""
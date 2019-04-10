from numpy import exp as exp
import numpy as np

class stellar_engine:
    def __init__(self):
        ### Constants and values defined here for faster re-initialization of objects in loops
        self.N_A = 6.022e23              # Avrogados number
        self.MeV = 1.602e-13             # MeV in Joule
        self.m_u = 1.66053904e-27        # Unit atomic mass
        ## Reaction-energies in Joule
        # Shared reaction
        self.Q_pp = (1.02 + 0.15 + 5.49)*self.MeV

        # PPI
        self.Q_33 = 12.86*self.MeV

        # PPII
        self.Q_34 = 1.59*self.MeV
        self.Q_e7 = 0.05*self.MeV
        self.Q_17_ = 17.35*self.MeV

        # PPIII
        self.Q_17 = (0.14 + 1.02 + 6.88 + 3.00)*self.MeV

        ## Mass fractions and particle numbers:
        self.X = 0.7; self.Y = 0.29; self.Z = 0.01
        self.Y_3 = 1e-10
        self.Z_Li = 1e-13; self.Z_Be = 1e-13

    def __call__(self,rho,T,project0 = False,sanity=False):
        T90 = T/1e9                 # Scaled temperatures used in lambdas below
        T91 = T90/(1+4.95e-2*T90)
        T92 = T90/(1+0.759*T90)
        
        if project0: # Used for sanity check from app. C
            print('Fractions of Lithium and Berilium from app. C used!')
            self.Z_Li = 1e-7; self.Z_Be = 1e-7

        ## Particle numbers:
        n_p = rho*self.X/self.m_u
        n_He3 = rho*self.Y_3/(3*self.m_u)
        n_He4 = rho*(self.Y-self.Y_3)/(4*self.m_u)
        n_Li7 = rho*self.Z_Li/(7*self.m_u)
        n_Be7 = rho*self.Z_Be/(7*self.m_u)
        #self.n_e = rho*(X + 2./3*Y_3+0.5*(Y-Y_3)+Z_Li/7+2/7*Z_Be)/m_u
        n_e = n_p+2*n_He3+2*n_He4+0.5*7*n_Li7+0.5*7*n_Be7    

        ## Proportion functions, e.i. lambdas:
        N_A_SI_convert = 1/self.N_A/1e6 # Scale tabulated values to lambdas in SI
        l_pp = N_A_SI_convert * (4.01e-15 * T90**(-2/3) * exp(-3.380*T90**(-1/3)) * (1+0.123*T90**(1/3)+1.09*T90**(2/3)+0.938*T90))

        l_33 = N_A_SI_convert * (6.04e10 * T90**(-2/3) * exp(-12.276*T90**(-1/3))*(1+0.034*T90**(1/3)-0.522*T90**(2/3)-0.124*T90+0.353*T90**(4/3)+0.213*T90**(-5/3)))

        l_34 = N_A_SI_convert * (5.61e6 * T91**(5/6)*T90**(-3/2) * exp(-12.826*T91**(-1/3)))

        l_e7 = N_A_SI_convert * (1.34e-10 * T90**(-1/2) * (1-0.537*T90**(1/3)+3.86*T90**(2/3)+0.0027*T90**(-1)*exp(2.515e-3*T90**(-1))))

        l_17_ = N_A_SI_convert * ( 1.096e9*T90**(-2/3.)*exp(-8.472*T90**(-1/3.)) -4.83e8*T92**(5/6)*T90**(-3/2)*exp(-8.472*T92**(-1/3)) + 1.06e10*T90**(-3/2)*exp(-30.442/T90) ) 

        l_17 = N_A_SI_convert * (3.11e5 * T90**(-2/3) * exp(-10.262*T90**(-1/3)) + 2.53e3 * T90**(-3/2) * exp(-7.306*T90**(-1)))

        if T < 1e6:    # Check if upper electron capture limit is needed
            if l_e7 > 1.57e-7/self.N_A/n_e:
                #print('Upper electron capture limit used!')
                l_e7 = 1.57e-7/self.N_A/n_e
                pass
        
        #print ('lambdas\n',self.l_pp,self.l_33,self.l_34,self.l_e7,self.l_17_,self.l_17)

        ## Initial production rate values
        # Base reaction
        r_pp = n_p**2 * l_pp / (2*rho)
        # PPI
        r_33 = n_He3**2 * l_33 / (2*rho)
        # PPII
        r_34 = n_He3*n_He4 * l_34 / rho
        r_e7 = n_e*n_Be7 * l_e7 / rho
        r_17_ = n_p*n_Li7 * l_17_ / rho
        # PPIII
        r_17 = n_p*n_Be7 * l_17 / rho

        # Available amounts from prior reactions checks
        r_33_34 = (2*r_33 + r_34)
        if r_33_34 > r_pp:
            K = r_pp/r_33_34
            r_33 = K*r_33
            r_34 = K*r_34

        r_e7_17 = (r_e7 + r_17)
        if r_e7_17 > r_34:
            K = r_34/r_e7_17
            r_e7 = K*r_e7
            r_17 = K*r_17

        if r_17_ > r_e7:
            K = r_e7/r_17_
            r_17_ = K*r_17_

        ## Energies
        e_pp = self.Q_pp*r_pp
        e_33 = self.Q_33*r_33
        e_34 = self.Q_34*r_34
        e_e7 = self.Q_e7*r_e7
        e_17_ = self.Q_17_*r_17_
        e_17 = self.Q_17*r_17
        #print(e_17,e_34+e_e7+e_17_,np.abs(e_17-(e_34+e_e7+e_17_)))
        # From each chain
        # self.energy_PP1 = r_33/(r_33+r_34)*e_pp + e_33                      #(2*self.Q_pp+self.Q_33)*r_33 #
        # self.energy_PP2 = r_34/(2*(r_34+r_33))*e_pp+e_34+e_e7+e_17_         #(self.Q_pp+self.Q_34)*r_34 + self.Q_e7*r_e7 + self.Q_17_*r_17_ 
        # self.energy_PP3 = r_34/(2*(r_34+r_33))*e_pp+e_17                    #(self.Q_pp+self.Q_34)*r_34 + self.Q_17*r_17#
        self.energy_PP1 = (2*self.Q_pp+self.Q_33)*r_33
        self.energy_PP2 = (self.Q_pp+self.Q_34)*r_34 + self.Q_e7*r_e7 + self.Q_17_*r_17_
        self.energy_PP3 = (self.Q_pp+self.Q_34)*r_34 + self.Q_17*r_17
        
        if sanity:
            return e_pp,e_33,e_34,e_e7,e_17_,e_17
        else:
            total_energy = e_pp+e_33+e_34+e_e7+e_17_+e_17
            # if np.abs(total_energy-(self.energy_PP1+self.energy_PP2+self.energy_PP3))>1e-6:
            #     print('Total energy not equal sum of PP chains')
            #     print(total_energy,self.energy_PP1+self.energy_PP2+self.energy_PP3,self.energy_PP1,self.energy_PP2,self.energy_PP3)
            return total_energy

def test_engine_1():

        goals = np.array([4.04e2,1.94e-6,4.88e-5,1.53e-6,5.29e-4,1.57e-6])
        
        rho = 1.62e5
        T = 1.57e7
        
        test = stellar_engine(rho,T,project0=True)
        results = np.asarray(test(sanity=True))*rho

        header = 'T = {:.2e}'.format(T)
        print('{:^35s}'.format(header))
        print(35*'-')

        print('{:^11s}|{:^11s}|{:^11s}'.format('Computed','Goal','Relative error'))
        for r,g in zip(results,goals):
            rel_error = np.abs((g-r)/g)
            print('{:^11.3e}|{:^11.3e}|{:^11.6f}'.format(r,g,rel_error))
        print(35*'-')

def test_engine_2():

    goals = np.array([7.33e4,1.3e1,1.72e4,1.26e-3,4.35e-1,1.21e5])

    rho = 1.62e5
    T = 10**8

    test = stellar_engine(rho,T,project0=True)
    results = np.asarray(test(sanity=True))*rho

    header = 'T = {:.2e}'.format(T)
    print('{:^35s}'.format(header))
    print(35*'-')

    print('{:^11s}|{:^11s}|{:^11s}'.format('Computed','Goal','Relative error'))
    for r,g in zip(results,goals):
        rel_error = np.abs((g-r)/g)
        print('{:^11.3e}|{:^11.3e}|{:^11.6f}'.format(r,g,rel_error))
    print(35*'-')

if __name__ == '__main__':
    test_engine_1()
    test_engine_2()


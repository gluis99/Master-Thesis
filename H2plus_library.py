#library for H2+ molecule
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
from sympy.physics.wigner import wigner_3j as w3j
from sympy.physics.wigner import wigner_6j as w6j
from sympy.physics.quantum.cg import CG
from scipy.linalg import eig
from scipy.sparse.linalg import eigs
from scipy import constants
from math import pi
from fractions import Fraction
from sympy import I

########################################################################################################################
#contants
#g_e = constants.physical_constants['electron g factor'][0]
g_e = 2.00231930436256 #Electron g-factor (minus sign is contained in the Hamiltonians below)
#g_p = constants.physical_constants['proton g factor'][0]
g_p = 5.5856946893 #Proton g-factor
µ_b = constants.physical_constants['Bohr magneton in Hz/T'][0]
#µ_b = 9.2740100783*10**(-24)  #Bohr magneton in J/T
µ_ratio = constants.physical_constants['proton mag. mom. to Bohr magneton ratio'][0]
µ_p = µ_ratio * µ_b
m_ratio_ep = constants.physical_constants['electron-proton mass ratio'][0]
m_ratio_pe = constants.physical_constants['proton-electron mass ratio'][0]
#µ_p = 5.0507837461*10**(-27) # Nuclear magneton in J/T
h = constants.physical_constants['Planck constant'][0]
#h = 6.62607015*10**(-34) #Planck constant in J/Hz
hbar = constants.hbar
m_e = constants.m_e
m_p = constants.m_p
c = constants.c
epsilon_0 = constants.epsilon_0
a_0 = constants.physical_constants['Bohr radius'][0]
e_charge = constants.e
alpha = constants.alpha

S_e = 0.5

########################################################################################################################
#Dissociation energies for the ro-vibrational states of the ground electronic state of H2+ (in MHz)

#from R. Moss, Molecular Physics 80, 1541 (1993) in reciprocal centimeters converted to MHz
def rovib_diss_energy(nu, L, relativistic=True):
    Moss_energies_nonrel = {
        (0, 0): 21379.29228, (0, 1): 21321.06033, (0,2): 21205.06068, (0,3): 21032.20981, (0,4): 20803.85310,
        (1, 0): 19188.19278, (1, 1): 19133.02604, (1,2): 19023.13742, (1,3): 18859.40504, (1,4): 18643.11770,
        (2, 0): 17124.30282, (2, 1): 17072.09631, (2,2): 16968.10991, (2,3): 16813.18564, (2,4): 16608.55915,
        (3, 0): 15183.39963, (3, 1): 15134.06383, (3,2): 15035.80168, (3,3): 14889.42129, (3,4): 14696.10844
    }
    #Relativistic and radiative corrections are included
    Moss_energies_rel = {
        (0, 0): 21379.3501, (0, 1): 21321.1164, (0,2): 21205.1133, (0,3): 21032.2574, (0,4): 20803.8941,
        (1, 0): 19188.2235, (1, 1): 19133.0551, (1,2): 19023.1634, (1,3): 18859.4263, (1,4): 18643.1328,
        (2, 0): 17124.3095, (2, 1): 17072.1015, (2,2): 16968.1122, (2,3): 16813.1836, (2,4): 16608.5514,
        (3, 0): 15183.3851, (3, 1): 15134.0480, (3,2): 15035.7831, (3,3): 14889.3988, (3,4): 14696.0807
    }
    
    if relativistic:
        if (nu, L) in Moss_energies_rel:
            return Moss_energies_rel[(nu, L)]*c*1e-4
    else:
        if (nu,L) in Moss_energies_nonrel:
            return Moss_energies_nonrel[(nu, L)]*c*1e-4
            
    print(f"Dissociation energy for (nu,L) = ({nu},{L}) not available.")
    return None

def Energy_diff(nu1, L1, nu2, L2, rel=True):
    return rovib_diss_energy(nu1, L1, rel) - rovib_diss_energy(nu2, L2, rel)

#From Horacio Olivares Pilón and Daniel Baye (2012) J. Phys. B: At. Mol. Opt. Phys. 45065101 (converted to MHz)
E_diss = -0.499727839716 * 6.5796839207293e9

########################################################################################################################
#get the coupling constants for the hyperfine Hamiltonian, where default is the new improved ones
def const_lit_fs(nu,L,new=True,print_which=True):
    #coupling constants in the effective hyperfine Hamiltonian (in MHz)

    def b_F(nu, L, take_new):
        #from Karr et al., PRA 102, 052827 (2020)
        data_2020 = {
            (0, 1): 922.9301, (1, 1): 898.7493, (2, 1): 876.3961, (3, 1): 855.7560, (4, 1): 836.7287,
            (5, 1): 819.2267, (6, 1): 803.1745, (7, 1): 788.5075, (8, 1): 775.1712, (9, 1): 763.1211, (10, 1): 752.3219,
            (0, 3): 917.5297, (1, 3): 893.6950, (2, 3): 871.6699, (3, 3): 851.3422, (4, 3): 832.6136,
            (5, 3): 815.3988, (6, 3): 799.6241, (7, 3): 785.2269, (8, 3): 772.1546, (9, 3): 760.3644, (10, 3): 749.8233
        }
        #from V. I. Korobov et al. (2016)
        data_2016 = {
            (0,1): 922.9318, (1,1): 898.7507, (2,1): 876.3973, (3,1): 855.7570, (4,1): 836.7294,
            (5,1): 819.2272, (6,1): 803.1750, (7,1): 788.5079, (8,1): 775.1714
        }
        #from Korobov et al. (2006)
        data_2006 = {
            (0, 1): 922.992, (1, 1): 898.809, (2, 1): 876.454, (3, 1): 855.812,
            (0, 3): 917.591, (1, 3): 893.755, (2, 3): 871.728, (3, 3): 851.398,
            (0, 5): 917.591, (0, 7): 917.591, (0, 9): 917.591
        }

        if L % 2 == 0:
            return 0
        elif (nu, L) in data_2020 and take_new:
            if print_which:
                print(f"New b_F({nu},{L}) from 2020")
            return data_2020[(nu, L)]
        elif (nu, L) in data_2006:
            if print_which:
                print(f"b_F({nu},{L}) from 2006")
            return data_2006[(nu, L)]
        else:
            print(f"b_F({nu},{L}) not available.")
            return None

    def c_e(nu, L, take_new):
        #from Haidar et al., PRA 106, 022816 (2022)
        data_2022 = {
            (0, 1): 42.41732, (4, 1): 32.65532, (5, 1): 30.43780, (6, 1): 28.28095,
            (0, 2): 42.16352, (1, 2): 39.57250
        }
        #from V. I. Korobov et al., Phys. Rev. A 102, 022840 (2020)
        data_2020 = {
            (0,2): 42.16399, (1,2): 39.57294
        }
        #from Korobov et al. (2006)
        data_2006 = {
            (0, 1): 42.4163, (1, 1): 39.8122, (2, 1): 37.3276, (3, 1): 34.9468,
            (0, 2): 42.1625, (1, 2): 39.5716, (2, 2): 37.0992, (3, 2): 34.7295,
            (0, 3): 41.7866, (1, 3): 39.2152, (2, 3): 36.7608, (3, 3): 34.4078,
            (0, 4): 41.2942, (1, 4): 38.7483, (2, 4): 36.3175, (3, 4): 33.9864,
            (0, 5): 41.2942, (0, 6): 41.2942, (0, 7): 41.2942, (0, 8): 41.2942, (0, 9): 41.2942
        }

        if L == 0:
            return 0
        elif (nu, L) in data_2022 and take_new:
            if print_which:
                print(f"New c_e({nu},{L}) from 2022")
            return data_2022[(nu, L)]
        elif (nu, L) in data_2006:
            if print_which:
                print(f"c_e({nu},{L}) from 2006")
            return data_2006[(nu, L)]
        else:
            print(f"c_e({nu},{L}) not available.")
            return None

    def c_I(nu, L):
        #from Korobov et al. (2006)
        data = {
            (0, 1): -4.168e-02, (1, 1): -4.035e-02, (2, 1): -3.893e-02, (3, 1): -3.742e-02,
            (0, 3): -4.076e-02, (1, 3): -3.944e-02, (2, 3): -3.803e-02, (3, 3): -3.654e-02,
            (0, 5): -4.076e-2, (0, 7): -4.076e-2, (0, 9): -4.076e-2
        }

        if L % 2 == 0:
            return 0
        elif (nu, L) in data:
            if print_which:
                print(f"c_I({nu},{L}) from 2006")
            return data[(nu, L)]
        else:
            print(f"c_I({nu},{L}) not available.")
            return None

    def d1(nu, L, take_new):
        #from Haidar et al., PRA 106, 022816 (2022)
        data_2022 = {
            (0, 1): 15*8.566174, (4, 1): 15*6.537386, (5, 1): 15*6.080400, (6, 1): 15*5.637627
        }
        #from Korobov et al. (2006)
        data_2006 = {
            (0, 1): 128.490, (1, 1): 120.337, (2, 1): 112.579, (3, 1): 105.169,
            (0, 3): 127.013, (1, 3): 118.940, (2, 3): 111.255, (3, 3): 103.910,
            (0, 5): 127.013, (0, 7): 127.013, (0, 9): 127.013
        }

        if L % 2 == 0:
            return 0
        elif (nu, L) in data_2022 and take_new:
            if print_which:
                print(f"New d1({nu},{L}) from 2022")
            return data_2022[(nu, L)]
        elif (nu, L) in data_2006:
            if print_which:
                print(f"d1({nu},{L}) from 2006")
            return data_2006[(nu, L)]
        else:
            print(f"d_1({nu},{L}) not available.")
            return None

    def d_2(nu, L):
        #from Korobov et al. (2006)
        data = {
            (0, 1): -0.2975, (1, 1): -0.2849, (2, 1): -0.2722, (3, 1): -0.2593,
            (0, 3): -0.2917, (1, 3): -0.2791, (2, 3): -0.2665, (3, 3): -0.2538,
            (0, 5): -0.2917, (0, 7): -0.2917, (0, 9): -0.2917
        }

        if L % 2 == 0:
            return 0
        elif (nu, L) in data:
            if print_which:
                print(f"d_2({nu},{L}) from 2006")
            return data[(nu, L)]
        else:
            print(f"d_2({nu},{L}) not available.")
            return None

    
    return b_F(nu,L,new), c_e(nu,L,new), c_I(nu,L), d1(nu,L,new), d_2(nu,L)

########################################################################################################################
#Hyperfine states for a given ro-vibrational state (nu,L) of H2+

def states_FJM(L,Karr_order=False):
    if Karr_order:
        return states_FJM_Karr(L)
    return states_FJM_myorder(L)

#Ordered as in Karr et al. PRA 77 (2008)
def states_FJM_Karr(L):
    """
    Sorts the states in decreasing order by J, then F, and finally MJ.

    Parameters
    ----------
    L :

    Returns
    -------
    ndarray
        The sorted states.
    """
    states = states_FJM_myorder(L)
    # Define a function that returns the sort keys
    def sort_key(state):
        F, J, MJ = state
        return (-J, -F, -MJ)

    # Sort the states using the sort keys
    sorted_states = sorted(states, key=sort_key)

    return np.array(sorted_states, dtype=Fraction)

#Ordered decreasingly first in F, then J, then MJ
def states_FJM_myorder(L):
    """
    Computes the set of states (F,J,MJ) with orbital angular momentum quantum number L.

    Parameters
    ----------
    L : int
        Orbital angular momentum quantum number.

    Returns
    -------
    ndarray
        The set of states (F,J,MJ) with orbital angular momentum quantum number L.
    """
    states = np.array([], dtype=Fraction)
    #for even L
    if L%2 == 0:
        F = Fraction(1, 2)
        Js = np.arange(abs(L-F),L+F+1, dtype=Fraction)
        nmJs = 2*Js+1
        for i in range(np.size(Js)):
            J = Js[i]
            for k in range(int(nmJs[i])):
                MJ = k-J
                state = np.array([F,J,MJ], dtype=Fraction)
                states = np.append(states, state)
    #for odd L
    else:
        Fs = np.array([Fraction(1, 2), Fraction(3, 2)], dtype=Fraction)
        for j in range(np.size(Fs)):
            F = Fs[j]
            Js = np.arange(abs(L-Fs[j]),L+Fs[j]+1, dtype=Fraction)
            for i in range(np.size(Js)):
                J = Js[i]
                nm = int(2*J+1)
                for k in range(nm):
                    MJ = k - J
                    state = np.array([F,J,MJ], dtype=Fraction)
                    states = np.append(states, state)
    states = np.reshape(states, (-1,3))
    return states

def states_FJ(L):
    """
    Computes the set of reduced states (F,J) with orbital angular momentum quantum number L.

    Parameters
    ----------
    L : int
        Orbital angular momentum quantum number.

    Returns
    -------
    ndarray
        The set of reduced states (F,J) with orbital angular momentum quantum number L.
    """
    states = np.array([], dtype=Fraction)
    #for even L
    if L%2 == 0:
        F = Fraction(1, 2)
        Js = np.arange(abs(L-F),L+F+1, dtype=Fraction)
        for k in range(np.size(Js)):   
            J = Js[k]
            state = np.array([F,J], dtype=Fraction)
            states = np.append(states, state)
    #for odd L
    else:
        Fs = np.array([Fraction(1, 2), Fraction(3, 2)], dtype=Fraction)
        for i in range(np.size(Fs)):
            F = Fs[i]
            Js = np.arange(abs(L-Fs[i]),L+Fs[i]+1, dtype=Fraction)
            for k in range(np.size(Js)):
                J = Js[k]
                state = np.array([F,J], dtype=Fraction)
                states = np.append(states, state)
    states = np.reshape(states, (-1,2))
    return states

def states_FJ_index(F, J, L):
    """
    Returns the index of the state with quantum numbers F and J in the array of states with
    orbital angular momentum quantum number L.

    Parameters
    ----------
    F : Fraction
        Total spin quantum number.
    J : Fraction
        Total angular momentum quantum number.
    L : int
        Orbital angular momentum quantum number.

    Returns
    -------
    int
        The index of the state with quantum numbers F and J in the array of states with
        orbital angular momentum quantum number L.
    """
    states = states_FJ(L)
    for i in range(len(states)):
        if states[i][0] == F and states[i][1] == J:
            return i
    return None

def states_FJM_index(F, J, MJ, L):
    """
    Returns the index of the state with quantum numbers F and J in the array of states with
    orbital angular momentum quantum number L.

    Parameters
    ----------
    F : Fraction
        Total spin quantum number.
    J : Fraction
        Total angular momentum quantum number.
    L : int
        Orbital angular momentum quantum number.

    Returns
    -------
    int
        The index of the state with quantum numbers F and J in the array of states with
        orbital angular momentum quantum number L.
    """
    states = states_FJM(L)
    for i in range(len(states)):
        if states[i][0] == F and states[i][1] == J and states[i][2] == MJ:
            return i
    return None

########################################################################################################################
#Hyperfine Hamiltonian for a given ro-vibrational state (nu,L) of H2+

def H_hf_red(nu,L):
    """
    Computes the reduced hyperfine Hamiltonian matrix in the basis (F,J) for a rovibrational level (nu,L)
    where Hhf(F,J;F',J') are the matrix elements

    Parameters
    ----------
    nu : int
        vibrational quantum number
    L : int
        orbital angular momentum quantum number

    Returns
    -------
    ndarray
        The reduced hyperfine Hamiltonian matrix in the basis (F,J).
    """
    states = states_FJ(L)
    N = len(states)
    
    Se = 1/2
    if L%2 == 0:
        I=0
    else:
        I=1
    
    ISe = np.zeros((N,N))
    LSe = np.zeros((N,N))
    LI = np.zeros((N,N))
    
    
    for i in range(N):
        for j in range(N):
            F1,J1 = states[i]
            F2,J2 = states[j]
            
            F1_, F2_ = float(F1), float(F2)
            
            if J1==J2:
                if F1==F2:
                    ISe[i,j] = 1/2*(F1*(F1+1)-Se*(Se+1)-I*(I+1))
                
                LSe[i,j] = (-1)**(J1+L+F1+F2+I+Se+1) *np.sqrt((2*F1_+1)*(2*F2_+1)) *w6j(L,1,L,F2,J1,F1)*w6j(Se,1,Se,F2,I,F1) \
                    *np.sqrt(L*(L+1)*(2*L+1)) *np.sqrt(Se*(Se+1)*(2*Se+1))
                
                LI[i,j] = (-1)**(J1+L+F1+F1+I+Se+1) *np.sqrt((2*F1_+1)*(2*F2_+1)) *w6j(L,1,L,F2,J1,F1)*w6j(I,1,I,F2,Se,F1) \
                    *np.sqrt(L*(L+1)*(2*L+1)) *np.sqrt(I*(I+1)*(2*I+1))
    
    #coupling constants
    bF, ce, cI, d1, d2 = const_lit_fs(nu,L)

    #matrix of the hyperfine Hamiltonian
    H = bF*ISe + ce*LSe + cI*LI + d1/((2*L-1)*(2*L+3))*((2/3)*L*(L+1)*ISe - LI.dot(LSe) - LSe.dot(LI)) \
                                + d2/((2*L-1)*(2*L+3))*((1/3)*L*(L+1)*I*(I+1)*np.eye(N) - 1/2*LI - LI.dot(LI))
    
    return H

def H_hf(nu,L):
    """
    Computes the hyperfine Hamiltonian matrix in the basis (F,J,MJ) for a rovibrational level (nu,L)
    where Hhf(F,J,MJ;F',J',MJ') are the matrix elements

    Parameters
    ----------
    nu : int
        vibrational quantum number
    L : int
        orbital angular momentum quantum number

    Returns
    -------
    ndarray
        The hyperfine Hamiltonian matrix in the basis (F,J,MJ) in MHz.
    """
    states = states_FJM(L)
    N = len(states)
    
    Se = 1/2
    if L%2 == 0:
        I=0
    else:
        I=1
    
    ISe = np.zeros((N,N))
    LSe = np.zeros((N,N))
    LI = np.zeros((N,N))
    
    
    for i in range(N):
        for j in range(N):
            F1,J1,M1 = states[i]
            F2,J2,M2 = states[j]
            
            F1_, F2_ = float(F1), float(F2)
            
            if J1==J2 and M1==M2:
                if F1==F2:
                    ISe[i,j] = 1/2*(F1*(F1+1)-Se*(Se+1)-I*(I+1))
                
                LSe[i,j] = (-1)**(J1+L+F1+F2+I+Se+1) *np.sqrt((2*F1_+1)*(2*F2_+1)) *w6j(L,1,L,F2,J1,F1)*w6j(Se,1,Se,F2,I,F1) \
                    *np.sqrt(L*(L+1)*(2*L+1)) *np.sqrt(Se*(Se+1)*(2*Se+1))
                
                LI[i,j] = (-1)**(J1+L+F1+F1+I+Se+1) *np.sqrt((2*F1_+1)*(2*F2_+1)) *w6j(L,1,L,F2,J1,F1)*w6j(I,1,I,F2,Se,F1) \
                    *np.sqrt(L*(L+1)*(2*L+1)) *np.sqrt(I*(I+1)*(2*I+1))
    
    
    #coupling constants
    bF, ce, cI, d1, d2 = const_lit_fs(nu,L)
    #different contributions to the Hamiltonian
    H1 = ISe
    H2 = LSe
    H3 = LI
    H4 = 2/3*L*(L+1)*ISe - LI.dot(LSe) - LSe.dot(LI)
    H5 = 1/3*L*(L+1)*I*(I+1)*np.eye(N) - 1/2*LI - LI.dot(LI)
    
    #matrix of the hyperfine Hamiltonian
    #H = bF * ISe + ce*LSe + cI*LI + d1/((2*L-1)*(2*L+3)) *(2/3*L*(L+1)*ISe - LI.dot(LSe) - LSe.dot(LI)) + d2/((2*L-1)*(2*L+3)) *(1/3*L*(L+1)*I*(I+1)*np.eye(N) - 1/2*LI - LI.dot(LI))
    H = bF*H1 + ce*H2 + cI*H3 + d1/((2*L-1)*(2*L+3))*H4 + d2/((2*L-1)*(2*L+3))*H5
    
    return H

########################################################################################################################
#Zeeman hamiltonian for a given ro-vibrational state (nu,L) of H2+

#Values of the matrix element <Ltot> = <nu,L|L_e|nu,L>/sqrt(2L+1) -2(m_e/m_p) <nu,L|L_p|nu,L>/sqrt(2L+1)  
#calculated using a variational approach (from Karr et al. PRA 77 062507 (2008))
def Ltot(nu, L):
    #from Karr et al. PRA 77 062507 (2008)
    data = {
        (0, 1): -0.7087e-03, (1, 1): -0.7015e-03, (2, 1): -0.6938e-03, (3, 1): -0.6855e-03, (4, 1): -0.6764e-03,
        (0, 2): -1.2271e-03, (1, 2): -1.2146e-03, (2, 2): -1.2012e-03, (3, 2): -1.1867e-03, (4, 2): -1.1710e-03,
        (0, 3): -1.7344e-03, (1, 3): -1.7167e-03, (2, 3): -1.6977e-03, (3, 3): -1.6776e-03, (4, 3): -1.6547e-03,
        (0, 4): -2.2375e-03, (1, 4): -2.2146e-03, (2, 4): -2.1898e-03, (3, 4): -2.1629e-03, (4, 4): -2.1339e-03
    }

    if L == 0:
        return 0
    elif (nu, L) in data:
        return data[(nu, L)]
    else:
        print(f"Ltot({nu},{L}) not available.")
        return None

#L_i_div(nu,L) = <nu,L|L_i|nu,L>/sqrt(2L+1)
def L_e(nu, L):
    data = {
        (0, 1): -0.615e-04, (1, 1): -0.686e-04, (2, 1): -0.763e-04, (3, 1): -0.847e-04, (4, 1): -0.937e-04,
        (0, 2): -1.069e-04, (1, 2): -1.193e-04, (2, 2): -1.328e-04, (3, 2): -1.473e-04, (4, 2): -1.630e-04,
        (0, 3): -1.521e-04, (1, 3): -1.698e-04, (2, 3): -1.889e-04, (3, 3): -2.095e-04, (4, 3): -2.318e-04,
        (0, 4): -1.980e-04, (1, 4): -2.209e-04, (2, 4): -2.457e-04, (3, 4): -2.725e-04, (4, 4): -3.015e-04
    }

    if L == 0:
        return 0
    elif (nu, L) in data:
        return data[(nu, L)] * np.sqrt(2*L+1)
    else:
        print(f"L_e({nu},{L}) not available.")
        return None  

def L_1(nu, L):
    data = {
        (0, 1): 0.70708, (1, 1): 0.70707, (2, 1): 0.70707, (3, 1): 0.70706, (4, 1): 0.70706,
        (0, 2): 1.22469, (1, 2): 1.22469, (2, 2): 1.22468, (3, 2): 1.22467, (4, 2): 1.22466,
        (0, 3): 1.73197, (1, 3): 1.73197, (2, 3): 1.73196, (3, 3): 1.73195, (4, 3): 1.73193,
        (0, 4): 2.23597, (1, 4): 2.23596, (2, 4): 2.23595, (3, 4): 2.23593, (4, 4): 2.23592
    }

    if L == 0:
        return 0
    elif (nu, L) in data:
        return data[(nu, L)] * np.sqrt(2*L+1)
    else:
        print(f"L_1({nu},{L}) not available.")
        return None

#The q-th canonical component of the dipole operator µ in MHz/T
def mu(q,nu,L):
    states = states_FJM(L)
    N = len(states)
    
    Se = 1/2
    if L%2 == 0:
        I=0
    else:
        I=1
    
    Mu = np.zeros((N,N))
    
    
    
    for i in range(N):
        for j in range(N):
            F1,J1,M1 = states[i]
            F2,J2,M2 = states[j]
            
            F1_, F2_ = float(F1), float(F2)
            J1_, J2_ = float(J1), float(J2)
            
            #factor from Wigner Eckert theorem to get reduced matrix element
            WE = CG(J2,M2,1,q,J1,M1).doit()
            
            #Spin contributions
            #Electron spin contribution
            red_Se = (-1)**(J1+L+F2+F2+Se+I) *w6j(F1,1,F2,J2,L,J1)*w6j(Se,1,Se,F2,I,F1) *np.sqrt(2*J2_+1) *np.sqrt((2*F1_+1)*(2*F2_+1)) *np.sqrt(Se*(Se+1)*(2*Se+1))
            #Nuclear spin contribution
            red_I = (-1)**(J1+L+F2+F1+Se+I) *w6j(F1,1,F2,J2,L,J1)*w6j(I,1,I,F2,Se,F1) *np.sqrt(2*J2_+1) *np.sqrt((2*F1_+1)*(2*F2_+1)) *np.sqrt(I*(I+1)*(2*I+1))
            
            #Orbital angular momentum contributions
            if F1==F2:
                #red_Le = (-1)**(J2+L+F1+1) *w6j(L,1,L,J2,F1,J1) *np.sqrt((2*J2_+1)*(2*L+1)) *L_e(nu,L)
                #red_Li = (-1)**(J2+L+F1+1) *w6j(L,1,L,J2,F1,J1) *np.sqrt((2*J2_+1)*(2*L+1)) *L_1(nu,L)
                red_Ltot = (-1)**(J2+L+F1+1) *w6j(L,1,L,J2,F1,J1) *np.sqrt((2*J2_+1)*(2*L+1)) *Ltot(nu,L)
            else:
                #red_Le = 0
                #red_Li = 0
                red_Ltot = 0
            
            #Mu[i,j] = -WE *(g_e*µ_b*red_Se -g_p*µ_b*m_ratio_ep*red_I +µ_b*red_Le -2*µ_p*red_Li)
            Mu[i,j] = WE *(-g_e*µ_b*red_Se +g_p*µ_b*m_ratio_ep*red_I -µ_b*red_Ltot)
    return Mu*1e-6

#The q-th canonical component of the dipole operator (µ/µ_b) in unitless form
def mu_div_mub(q,nu,L):
    states = states_FJM(L)
    N = len(states)
    
    Se = 1/2
    if L%2 == 0:
        I=0
    else:
        I=1
    
    Mu = np.zeros((N,N))
    
    for i in range(N):
        for j in range(N):
            F1,J1,M1 = states[i]
            F2,J2,M2 = states[j]
            
            F1_, F2_ = float(F1), float(F2)
            J1_, J2_ = float(J1), float(J2)
            
            #factor from Wigner Eckert theorem to get reduced matrix element
            WE = CG(J2,M2,1,q,J1,M1).doit()
            
            #Spin contributions
            #Electron spin contribution
            red_Se = (-1)**(J1+L+F2+F2+Se+I) *w6j(F1,1,F2,J2,L,J1)*w6j(Se,1,Se,F2,I,F1) *np.sqrt(2*J2_+1) *np.sqrt((2*F1_+1)*(2*F2_+1)) *np.sqrt(Se*(Se+1)*(2*Se+1))
            #Nuclear spin contribution
            red_I = (-1)**(J1+L+F2+F1+Se+I) *w6j(F1,1,F2,J2,L,J1)*w6j(I,1,I,F2,Se,F1) *np.sqrt(2*J2_+1) *np.sqrt((2*F1_+1)*(2*F2_+1)) *np.sqrt(I*(I+1)*(2*I+1))
            
            #Orbital angular momentum contributions
            if F1==F2:
                red_Ltot = (-1)**(J2+L+F1+1) *w6j(L,1,L,J2,F1,J1) *np.sqrt((2*J2_+1)*(2*L+1)) *Ltot(nu,L)
            else:
                red_Ltot = 0
            
            Mu[i,j] = -WE *(g_e*red_Se -g_p*m_ratio_ep*red_I +red_Ltot)
    return Mu

def normHz(nu,L,printnonzeros=False):
    #get the states (F,J,MJ)
    states = states_FJM(L)
    N = len(states)
    if L%2==0:
        I=0
    else:
        I=1
    
    H = np.zeros((N,N))
    
    for i in range(N):
        F1_,J1_,MJ1 = states[i]
        F1 = float(F1_)
        J1 = float(J1_)
        for j in range(N):
            F2_,J2_,MJ2 = states[j]
            F2 = float(F2_)
            J2 = float(J2_)
            
            #factor from Wigner Eckert theorem to get reduced matrix element
            WE = (-1)**(J1-MJ1)*w3j(J1, 1, J2, -MJ1, 0, MJ2)
            
            #factor from the (L,F)J couplig
            #for F obs.: <((nu)L,(I,S_e)F1)J1||OF||(((nu)L,(I,S_e)F2)J2)> = C_F * <(I,S_e)F1||OF||(I,S_e)F2>
            C_F = (-1)**(J1_+L+F2_+1) *np.sqrt((2*J1+1)*(2*J2+1)) *w6j(F1_, 1, F2_, J2_, L  , J1_)
            #for L obs.: <((nu)L,(I,S_e)F1)J1||OL||(((nu)L,(I,S_e)F2)J2)> = C_L * <nu,L||OL||nu,L>
            C_L = (-1)**(J2_+L+F1_+1) *np.sqrt((2*J1+1)*(2*J2+1)) *w6j(L  , 1, L  , J2_, F1_, J1_)
            
            #factor from the (I,S_e)F coupling
            C_S = (-1)**(F2_+I+S_e+1) *np.sqrt((2*F1+1)*(2*F2+1)) *w6j(S_e, 1, S_e, F2_, I  , F1_)
            C_I = (-1)**(F1_+I+S_e+1) *np.sqrt((2*F1+1)*(2*F2+1)) *w6j(I  , 1, I  , F2_, S_e, F1_)
            
            #term corresponding to: +g_e*mu_b*(S_e)_z
            T1 =  g_e    *µ_b        *WE *C_F *C_S *np.sqrt(S_e*(S_e+1)*(2*S_e+1))
            #term corresponding to: -g_p*mu_p*I_z
            T2 = -g_p*µ_b*m_ratio_ep *WE *C_F *C_I *np.sqrt(I  *(I  +1)*(2*I  +1))
            #term corresponding to: +mu_b*(L_e)_z - mu_p*(L_1 + L_2)_z
            if F1==F2:
                T3 = µ_b             *WE *C_L *Ltot(nu,L)*np.sqrt(2*L+1)
            else:
                T3 = 0

            H[i,j] = T1 + T2 + T3
            
            if printnonzeros:
                if H[i,j] != 0:
                    print ("++++++++++++++++++++++++++++++++++++++++++++++")
                    print("i,j = ",i," , ",j)
                    print("F1 =",F1,",","J1 = ",J1, "MJ1 = ",MJ1)
                    print("F2 =",F2,",","J2 = ",J2, "MJ2 = ",MJ2)
                    print(H[i,j])
    
    return H*10**(-6)

########################################################################################################################
#Computes full Hamiltonian of the hyperfine structure of a rovibrational level (nu,L) in the (F,J,MJ) basis (in MHz)
#for the magnetic field B (in T)
def Htot(nu,L,B):
    H = H_hf(nu,L) + B*normHz(nu,L)
    return H

#Computes the Hamiltonian for a rovibrational level (nu,L) of H2+ 
#given the hyperfine Hamiltonian, the Zeeeman Hamiltonian and the magnetic field
def H_nuL(Hhf,Hz,B):
    return Hhf+B*Hz

########################################################################################################################
#States of the full Hamiltonian for a certain magnetic field B

def order_states_wprev(eigenvectors_previous, eigenvalues_current, eigenvectors_current, threshold=0.1):
    """
    Orders the eigenvalues to match the previous ones. This function can have problems for very small fields
    and may exchange degenerate states.
    It calculates the similarity between the current and previous eigenvectors, and if the similarity is
    above a certain threshold, it orders the eigenvalues and eigenvectors accordingly.

    Parameters
    ----------
    eigenvectors_previous : ndarray
        The eigenvectors from the previous calculation.
    eigenvalues_current : ndarray
        The current eigenvalues.
    eigenvectors_current : ndarray
        The current eigenvectors.
    threshold : float, optional
        The similarity threshold for ordering the eigenvalues and eigenvectors. Default is 0.1.


    Returns
    -------
    ordered_eigenvalues : ndarray
        The ordered eigenvalues.
    ordered_eigenvectors : ndarray
        The ordered eigenvectors.
    """
    N_ = len(eigenvalues_current)
    permutation = np.zeros(N_, dtype=int)
    used_indices = set()
    
    #print(np.abs(np.vdot(eigenvectors_current[:, 6], eigenvectors_previous[:, 0])))
    
    for i in range(N_):
        best_match = None
        best_match_index = -1

        for j in range(N_):
            if j not in used_indices:
                similarity = np.abs(np.vdot(eigenvectors_current[:, i], eigenvectors_previous[:, j]))
                if similarity > threshold and (best_match is None or similarity > best_match):
                    best_match = similarity
                    best_match_index = j
                    #print(similarity)
            
        #print(i,best_match_index)
        
        if best_match_index != -1:
            permutation[best_match_index] = i
            used_indices.add(best_match_index)
        else:
            print(i)
            raise ValueError("No good match found")
    
    
    
    ordered_eigenvalues = eigenvalues_current[permutation]
    ordered_eigenvectors = eigenvectors_current[:, permutation]
    
    return ordered_eigenvalues, ordered_eigenvectors

#For a given rovibrational level (nu,L), calculates the eigenvalues and eigenvectors
#of the total Hamiltonian for a given magnetic field B0, only correct order for small mixing
def get_EWs_EVs_B0(nu,L,B0):
    H = Htot(nu,L,B0)
    N = len(H)

    EWs_wo,EVs_wo = np.linalg.eigh(H)

    Basis_pure = np.eye(N)
    
    EWs = np.zeros(N)
    EVs = np.zeros((N,N))

    #limiting B field to change approach of ordering
    #chosen not too small, but small enough for all implemented (nu,L)
    Bchange=1e-9
    
    #get the eigenvalues in the right order
    #only consider biggest entry of EV to order (compare EVs to identity matrix)
    EWs, EVs = order_states_wprev(Basis_pure,EWs_wo,EVs_wo)


    return EWs, EVs

def get_EWs_B0(nu,L,B0):
    return get_EWs_EVs_B0(nu,L,B0)[0]

#For a given rovibrational level (nu,L), calculates the eigenvalues, eigenvectors and derivatives
#of the total Hamiltonian for a given set of magnetic field values Bvalues
def get_EWs_EVs_derivatives(nu,L,Bvalues):
    """
    Calculates the eigenvalues, eigenvectors and derivatives of the Hamiltonian for a given L and nu level.

    Parameters
    ----------
    L_level : int
        The L level.
    nu_level : int
        The nu level.
    Ham : function
        The Hamiltonian function for the rovibrational level L,nu.
    Bvalues : ndarray
        The values of the magnetic field.

    Returns
    -------
    eigenvalues : ndarray
        The eigenvalues.
    eigenvectors : ndarray
        The eigenvectors.
    derivatives : ndarray
        The derivatives.
    """
    #Get the hyperfine and Zeeman Hamiltonians
    matrix_Hhf = H_hf(nu,L)
    matrix_Hz = normHz(nu,L)

    N_ = len(matrix_Hhf)

    # Initialize empty lists to store all the eigenvalues and eigenvectors for each B
    EWs = []
    EVs_all = []
    for i in range(N_):
        EWs.append([])
        EVs_all.append([])

    EVs_prev = np.eye(N_)

    # Loop through each value of B
    for Bi in Bvalues:
        # Calculate the hamiltonian matrix
        Htot = H_nuL(matrix_Hhf,matrix_Hz,Bi)
    
        # Calculate the eigenvalues of H using np.linalg.eigh potentially in the wrong order
        EWs_wo_B,EVs_wo_B = np.linalg.eigh(Htot)
    
        EWs_B = np.zeros(N_)
        EVs_B = np.zeros((N_,N_))
    
        #limiting B field to change approach of ordering
        #chosen not too small, but small enough for all implemented (nu,L)
        Bchange=1e-9
    
        #get the eigenvalues in the right order
        #naive approach, assuming the mixing is small (for L=1it is for B<588µT)
        #only consider biggest entry of EV to order (compare EVs to identity matrix)
        if Bi<=Bchange:
            for i in range(N_):
                i_ordered = np.argmax(abs(EVs_wo_B[:,i]))
                EWs_B[i_ordered] = EWs_wo_B[i]
                EVs_B[:,i_ordered] = EVs_wo_B[:,i]
            #assure all eigenvalues are taken into account
            #print if there is a EWs_B[i] that is =0 and there are no EWs_wo_B[i] =0
            for i in range(N_):
                if EWs_B[i]==0:
                    for j in range(N_):
                        if EWs_wo_B[j]==0:
                            break
                    print("EWs_B[i] = 0 and EWs_wo_B[j] != 0 so something went wrong")
    
        #When the degeneracy in MJ is lifted enough this approach is more exact
        #Compare EVs at B to the EVs with the previous B
        else:
            EWs_B, EVs_B = order_states_wprev(EVs_prev,EWs_wo_B,EVs_wo_B)
            EVs_prev = EVs_B
    
        #append EVs at B to the list of all eigenvalues and eigenvectors
        for i in range(N_):
            EWs[i].append(EWs_B[i])
            EVs_all[i].append(EVs_B[:,i])

    # Convert the list of eigenvalues and eigenvectors to numpy arrays
    #`eigenvalues[i][j]` gives you the eigenvalue for the `i`-th eigenstate at the `j`-th B field value.
    eigenvalues = np.array(EWs)
    #`eigenvectors[i][j]` gives you the eigenvector for the `i`-th eigenstate at the `j`-th B field value
    #`eigenvectors[i][j][k]` gives you the `k`-th component of that eigenvector.
    eigenvectors = np.array(EVs_all)

    # Calculate the derivatives
    derivatives = np.gradient(eigenvalues, Bvalues, axis=1)

    return eigenvalues, eigenvectors, derivatives

#For given Hamiltonians Hhf and Hz, calculates the eigenvalues, eigenvectors and derivatives
def get_EWs_EVs_derivatives_Ham(matrix_Hhf,matrix_Hz,Bvalues):
    N_ = len(matrix_Hhf)

    # Initialize empty lists to store all the eigenvalues and eigenvectors for each B
    EWs = []
    EVs_all = []
    for i in range(N_):
        EWs.append([])
        EVs_all.append([])

    EVs_prev = np.eye(N_)

    # Loop through each value of B
    for Bi in Bvalues:
        # Calculate the hamiltonian matrix
        Htot = H_nuL(matrix_Hhf,matrix_Hz,Bi)
    
        # Calculate the eigenvalues of H using np.linalg.eigh potentially in the wrong order
        EWs_wo_B,EVs_wo_B = np.linalg.eigh(Htot)
    
        EWs_B = np.zeros(N_)
        EVs_B = np.zeros((N_,N_))
    
        #limiting B field to change approach of ordering
        #chosen not too small, but small enough for all implemented (nu,L)
        Bchange=1e-9
    
        #get the eigenvalues in the right order
        #naive approach, assuming the mixing is small (for L=1it is for B<588µT)
        #only consider biggest entry of EV to order (compare EVs to identity matrix)
        if Bi<=Bchange:
            for i in range(N_):
                i_ordered = np.argmax(abs(EVs_wo_B[:,i]))
                EWs_B[i_ordered] = EWs_wo_B[i]
                EVs_B[:,i_ordered] = EVs_wo_B[:,i]
            #assure all eigenvalues are taken into account
            #print if there is a EWs_B[i] that is =0 and there are no EWs_wo_B[i] =0
            for i in range(N_):
                if EWs_B[i]==0:
                    for j in range(N_):
                        if EWs_wo_B[j]==0:
                            break
                    print("EWs_B[i] = 0 and EWs_wo_B[j] != 0 so something went wrong")
    
        #When the degeneracy in MJ is lifted enough this approach is more exact
        #Compare EVs at B to the EVs with the previous B
        else:
            EWs_B, EVs_B = order_states_wprev(EVs_prev,EWs_wo_B,EVs_wo_B)
            EVs_prev = EVs_B
    
        #append EVs at B to the list of all eigenvalues and eigenvectors
        for i in range(N_):
            EWs[i].append(EWs_B[i])
            EVs_all[i].append(EVs_B[:,i])

    # Convert the list of eigenvalues and eigenvectors to numpy arrays
    #`eigenvalues[i][j]` gives you the eigenvalue for the `i`-th eigenstate at the `j`-th B field value.
    eigenvalues = np.array(EWs)
    #`eigenvectors[i][j]` gives you the eigenvector for the `i`-th eigenstate at the `j`-th B field value
    #`eigenvectors[i][j][k]` gives you the `k`-th component of that eigenvector.
    eigenvectors = np.array(EVs_all)

    # Calculate the derivatives
    derivatives = np.gradient(eigenvalues, Bvalues, axis=1)

    return eigenvalues, eigenvectors, derivatives

#For a given rovibrational level (nu,L), calculates the eigenvalues, eigenvectors and derivatives
#of the total Hamiltonian for a given value of the magnetic field B0
def get_EWs_EVs_derivatives_B0(nu,L,B0,deltaB=1e-7):
    Bvalues = np.linspace(B0-deltaB,B0+deltaB,100)
    EWs, EVs, dE_B = get_EWs_EVs_derivatives(nu, L, Bvalues)
    EWs_B0 = EWs[:,50]
    EVs_B0 = EVs[:,50]
    dE_B0 = dE_B[:,50]
    return EWs_B0, EVs_B0, dE_B0

########################################################################################################################
#Transition frequencies as a matrix for all the hyperfine levels of the rovibrational levels in MHz
def mat_transition_freqs_B(nu_i, Li, nu_f, Lf, B):
    if (nu_i, Li) == (nu_f, Lf):
        E_hf = get_EWs_B0(nu_i, Li, B)
        return E_hf - np.array(E_hf)[:, np.newaxis]
    else:
        E_hf_i = np.array(get_EWs_B0(nu_i, Li, B))[:, np.newaxis]  # Convert to column vector
        E_hf_f = get_EWs_B0(nu_f, Lf, B)
        #E_diff_levels = rovib_diss_energy(nu_f, Lf) - rovib_diss_energy(nu_i, Li)
        return E_hf_f - E_hf_i

def mat_transition_freqs_E(nu_i, Li, EWs_i, nu_f, Lf, EWs_f):
    if (nu_i, Li) == (nu_f, Lf):
        return EWs_f -  np.array(EWs_i)[:, np.newaxis]
    else:
        #E_diff_levels = rovib_diss_energy(nu_f, Lf) - rovib_diss_energy(nu_i, Li)
        return EWs_f - np.array(EWs_i)[:, np.newaxis]

#Transition wavelengths as a matrix for all the hyperfine levels of the rovibrational levels in m
def mat_transition_wavelengths_E(nu_i, Li, EWs_i, nu_f, Lf, EWs_f):
    return c / (mat_transition_freqs_E(nu_i, Li, EWs_i, nu_f, Lf, EWs_f)*1e6)

def mat_transition_sensitivities_B(nu_i, Li, nu_f, Lf, B):
    np.set_printoptions(suppress=True)
    _,_,DE_i = get_EWs_EVs_derivatives_B0(nu_i, Li, B)
    _,_,DE_f = get_EWs_EVs_derivatives_B0(nu_f, Lf, B)
    Ni,Nf = len(DE_i),len(DE_f)
    sensitivities = np.array([[DE_f[f]-DE_i[i] for f in range(Nf)] for i in range(Ni)])
    return np.round(sensitivities,4)

########################################################################################################################
#The reduced matrix element <nu_g,Lg||Q^(k)||nu_e,Le>, numerical results

def redQ(nu_g,Lg,nu_e,Le,k):
    nu_L_g = nu_g, Lg
    nu_L_e = nu_e, Le


    #from L Hilico et al 2001 J. Phys. B: At. Mol. Opt. Phys. 34 491, taken the sqrt of the Trans. prob. in table 3
    Q0_2001 = {
        ((0,0), (1,0)): -0.41857,
        ((0,0), (2,0)): -0.01763,
        ((0,0), (3,0)): -0.0006263,
    }
    #from Karr et al. PRA 77, 063410 (2008)
    Q0_2008 = {
        ((0, 0), (1, 0)): 0.7255,
        ((0, 1), (1, 1)): 1.261,
        ((0, 2), (1, 2)): 1.640,
        ((0, 3), (1, 3)): 1.962
    }
    Q2_2008 = {
        ((0, 1), (1, 1)): 0.7753,
        ((0, 2), (1, 2)): 0.8541,
        ((0, 3), (1, 3)): 0.9903
    }
    #from Karr H2+ and HD+: candidates for a molecular clock (2014)
    Q0_2014 = {
        ((0, 0), (1, 0)): -0.4189,
        ((0, 1), (1, 1)): -0.4204,
        ((0, 2), (1, 2)): -0.4234,
        ((0, 3), (1, 3)): -0.4280,
    }
    Q2_2014 = {
        ((0, 0), (1, 0)): 0,
        ((0, 0), (1, 2)): 0.5536,
        ((0, 1), (1, 1)): 0.3655,
        ((0, 1), (1, 3)): 0.3917,
        ((0, 2), (1, 0)): 0.2705,
        ((0, 2), (1, 2)): 0.3119,
        ((0, 2), (1, 4)): 0.3787,
        ((0, 3), (1, 1)): 0.3071,
        ((0, 3), (1, 3)): 0.3056
    }
    

    if k == 0:
        if Lg != Le:
            return 0
        elif (nu_L_g, nu_L_e) in Q0_2008:
            return Q0_2008[(nu_L_g, nu_L_e)]
        elif (nu_L_g, nu_L_e) in Q0_2014:
            return -Q0_2014[(nu_L_g, nu_L_e)]*np.sqrt(3)*np.sqrt(2*Lg+1)

        
    elif k == 2:
        if (nu_L_g, nu_L_e) in Q2_2008:
            return Q2_2008[(nu_L_g, nu_L_e)]
        elif (nu_L_g, nu_L_e) in Q2_2014:
            return Q2_2014[(nu_L_g, nu_L_e)]*np.sqrt(3/2)*np.sqrt(2*Lg+1)
    else:
        print(f"redQ({nu_L_g},{nu_L_e},{k}) not available.")
        return None

########################################################################################################################
#Calculations for 2 photon transitions

#matrix computing the elements <i|^(S)Q_(q1,q2)|f> for all pure states,
#where the initial state is i and the final state f
def mat_SQ(q1,q2,nu_g,Lg,nu_e,Le):
    """
    This function calculates the matrix elements of the operator ^(S)Q_(q1,q2)
    between all pairs of pure states |F,J,MJ>,where the initial state is i and the final state f.
    The operator ^(S)Q_(q1,q2) is a spherical tensor operator of rank k,
    q1 and q2 are the polarizations of the two photons in the standard polarization basis.

    Parameters
    ----------
    q1 : int
        The polarization of the first photon.
    q2 : int
        The polarization of the second photon.
    nu_g : float
        The vibrational quantum number of the ground state.
    Lg : int
        The total orbital angular momentum quantum number of the ground state.
    nu_e : float
        The vibrational quantum number of the excited state.
    Le : int
        The total orbital angular momentum quantum number of the excited state.

    Returns
    -------
    matrix : ndarray
        The matrix of elements <i|^(S)Q_(q1,q2)|f>.
    """
    states_g = states_FJM(Lg)
    if Lg%2 ==0:
        Ig = 0
    else:
        Ig = 1
    states_e = states_FJM(Le)
    if Le%2 ==0:
        Ie = 0
    else:
        Ie = 1
    Ng = len(states_g)
    Ne = len(states_e)
    def matrix_Q(k):
        matrix_Qk = np.zeros((Ng,Ne))
        for i in range(Ng):
            for f in range(Ne):
                #get the quantum numbers of the states i and f
                Fg,Jg,MJg = states_g[i]
                Fe,Je,MJe = states_e[f]
                Fg_,Jg_,Je_,MJg_=float(Fg),float(Jg),float(Je),float(MJg)
                if Ig == Ie and Fg == Fe:
                    redQk_Js = (-1)**(Je_+Lg+Fg_+k) *np.sqrt((2*Je_+1)) *w6j(Lg,k,Le,Je,Fg,Jg) *redQ(nu_g,Lg,nu_e,Le,k)
                    matrix_Qk[i,f] = CG(Je,MJe,k,q1+q2,Jg,MJg).doit() *redQk_Js
        return matrix_Qk
    #<i|^(S)Q_(q1,q2)|f> = sum_k a^k_(q1+q2) * <Je,MJe,k,q|Jg,MJg> *<(Lg,Fg)Jg||Q^(k)||(Le,Fe)Je>/sqrt(2Jg+1)
    #a^k_q = 
    a0_q = CG(1,q1,1,q2,0,q1+q2).doit()
    a2_q = CG(1,q1,1,q2,2,q1+q2).doit()

    return (a0_q * matrix_Q(0) + a2_q * matrix_Q(2)).astype(float)

#computes the matrix elements for the energy eigenstates of the Hamiltonian, which are mixed states
def mat_SQ_mix(q1,q2,nu_i,Li,EVs_i,nu_f,Lf,EVs_f):
    """
    Computes the matrix elements for the energy eigenstates of the Hamiltonian, which are mixed states.
    The state of (nu,L) is given by: sum_(F,J,MJ) c_(F,J,MJ) |F,J,MJ>.

    Parameters
    ----------
    q1 : int
        The polarization of the first photon.
    q2 : int
        The polarization of the second photon.
    nu_i : float
        The vibrational quantum number of the initial state.
    Li : int
        The total orbital angular momentum quantum number of the initial state.
    EVs_i : ndarray
        Array containing all the c_(F,J,MJ) coefficients of the initial eigenstates.
    nu_f : float
        The vibrational quantum number of the final state.
    Lf : int
        The total orbital angular momentum quantum number of the final state.
    EVs_f : ndarray
        Array containing all the c_(F,J,MJ) coefficients of the final eigenstates.

    Returns
    -------
    matrix : ndarray
        The matrix of elements for the energy eigenstates of the Hamiltonian.
    """
    
    Ni = len(EVs_i)
    Nf = len(EVs_f)
    matrix = np.zeros((Ni,Nf))
    
    #calculate matrix elements for the pure states
    SQ_matrix = mat_SQ(q1,q2,nu_i,Li,nu_f,Lf)
    
    for i in range(Ni):
        for f in range(Nf):
            EV_i = EVs_i[i]
            EV_f = EVs_f[f]

            for j in range(Ni):
                for k in range(Nf):
                    matrix[i,f] += EV_i[j]*EV_f[k] *SQ_matrix[j,k]
    matrix = np.where(np.abs(matrix) < 1e-16, 0, matrix)
    return matrix

#calculates the square of the transition matrix for a given transition
#from (nu_i,Li) to (nu_f,Lf) at a magnetic field B
def Qsquared(B,nu_i,Li,nu_f,Lf,q1,q2,print_pure=False,print_Qsquared=False,prec=8):
    """
    This function calculates the square of the 2 photon transition matrix
    for a given transition at a magnetic field B.

    Parameters
    ----------
    B : float
        The magnetic field strength.
    nu_i, Li, nu_f, Lf : int
        The initial and final ro-vibrational quantum numbers.
    q1, q2 : int
        The polarizations of the 2 photons driving the transition.
    print_pure : bool, optional
        If True, the function also prints the pure transition matrix. Default is False.
    prec : int, optional
        The number of digits of precision for the output. Default is 5.

    Returns
    -------
    Qsquared : ndarray
        The square of the 2 photon transition matrix.
    """
    # Calculate the energy states at the given magnetic field strength for the initial and final states
    EWs_i,EVs_i = get_EWs_EVs_B0(nu_i,Li,B)
    EWs_f,EVs_f = get_EWs_EVs_B0(nu_f,Lf,B)
    
    # Set the print options for numpy
    np.set_printoptions(suppress=True, precision=prec)
    
    # If print_pure is True, calculate and print the pure transition matrix
    if print_pure:
        pure_mat = mat_SQ(q1,q2,nu_i,Li,nu_f,Lf)
        print(pure_mat)
    
    # Calculate the mixed transition matrix
    Transition_mat = mat_SQ_mix(q1,q2,nu_i,Li,EVs_i,nu_f,Lf,EVs_f)

    # Square the transition matrix
    Qsquared = np.square(Transition_mat)
    
    # If print_Qsquared is True, print the squared transition matrix
    if print_Qsquared:
        print(Qsquared)
    
    return Qsquared

def averaged_Qsquared(B,nu_i,Li,nu_f,Lf,q1,q2):
    """
    This function calculates the square of the 2 photon transition matrix
    for a given transition at a magnetic field B and averages it over the magnetic sublevels.

    Parameters
    ----------
    B : float
        The magnetic field strength.
    nu_i, Li, nu_f, Lf : int
        The initial and final ro-vibrational quantum numbers.
    q1, q2 : int
        The polarizations of the 2 photons driving the transition.

    Returns
    -------
    Qsquared_avg : float
        The average of the square of the 2 photon transition matrix over the magnetic sublevels.
    """
    states_red_i = states_FJ(Li)
    states_red_f = states_FJ(Lf)
    N_red_i = len(states_red_i)
    N_red_f = len(states_red_f)
    
    Q_squared_matrix = Qsquared(B,nu_i,Li,nu_f,Lf,q1,q2) #for the eigenstates of the Hamiltonian which can be mixed
    #Q_squared_matrix = np.square(mat_SQ(q1,q2,nu_i,Li,nu_f,Lf)) #for the pure hyperfine states neglegting mixing
    
    
    avg_Q_squared_matrix = np.zeros((N_red_i,N_red_f))

    for i in range(N_red_i):
        for f in range(N_red_f):
            Fi, Ji = states_red_i[i]
            Ff, Jf = states_red_f[f]
            
            ni = int(2*Ji + 1)
            nf = int(2*Jf + 1)
            for ki in range(ni):
                for kf in range(nf):
                    MJi = Ji - ki
                    MJf = Jf - kf
                    avg_Q_squared_matrix[i,f] += Q_squared_matrix[states_FJM_index(Fi,Ji,MJi,Li),states_FJM_index(Ff,Jf,MJf,Lf)]/((2*Ji+1))
            #print('(Fi,Ji)=({0},{1}), (Ff,Jf)=({2},{3})'.format(Fi,Ji,Ff,Jf))
            #print('avg_Q^2 = ',avg_Q_squared_matrix[i,f])
    return avg_Q_squared_matrix

########################################################################################################################
#spherical basis vectors
def c_1(q):
    if q == 1:
        return -np.array([1,-1j,0])/np.sqrt(2)
    elif q == 0:
        return np.array([0,0,1])
    elif q == -1:
        return np.array([1,1j,0])/np.sqrt(2)

#second rank spherical tensors
def c_2(q):
    c_matrix = np.zeros((3,3), dtype=complex)
    norm_factor = 1/np.sqrt(6)
    if q==2:
        c_matrix[0,0],c_matrix[1,1] = 1,-1
        c_matrix[0,1],c_matrix[1,0] = -1j,-1j
    elif q==1:
        c_matrix[0,2],c_matrix[2,0] = -1,-1
        c_matrix[1,2],c_matrix[2,1] = 1j,1j
    elif q==0:
        c_matrix[0,0],c_matrix[1,1],c_matrix[2,2] = -1,-1, 2
        norm_factor = 1/3
    elif q==-1:
        c_matrix[0,2],c_matrix[2,0] = 1,1
        c_matrix[1,2],c_matrix[2,1] = 1j,1j
    elif q==-2:
        c_matrix[0,0],c_matrix[1,1] = 1,-1
        c_matrix[0,1],c_matrix[1,0] = 1j,1j
    return norm_factor*c_matrix

n_0 = np.array([np.sin(pi/4),0,np.cos(pi/4)])
eps_0 = np.array([np.cos(pi/4),0,-np.sin(pi/4)])
n_pm2 = np.array([np.sin(pi/2),0,np.cos(pi/2)])
eps_pm2 = np.array([0,1,0])
n_pm1 = np.array([1,0,0])
eps_pm1 = np.array([0,0,1])

#g_0 = abs(np.einsum('i,ij,j->', eps_0, c_2(0), n_0))
#g_pm2 = abs(np.einsum('i,ij,j->', eps_pm2, c_2(-2), n_pm2))

def spherical_components_vector(v):
    if len(v) != 3:
        print("The vector must have 3 components.")
        return
    #vector in spherical components (v_{-1}, v_0, v_{+1})
    return np.array([(v[0]-1j*v[1])/np.sqrt(2),v[2],-(v[0]+1j*v[1])/np.sqrt(2)])

########################################################################################################################
#Numerical results for the quadrupole coupling coefficient E_14 from D. Bakalov and S. Schiller, Appl. Phys. B 114, 213 (2014)
def E14(nu, L,in_au=False):
    #in MHz m^2 /GV
    data = {
        (0, 0): -0.0003018, (1, 0): -0.0003448, (2, 0): -0.000391, (3, 0): -0.0004409, (4, 0): -0.0004948, (5, 0): -0.0005533,
        (0, 1): 0.0001815, (1, 1): 0.0002074, (2, 1): 0.0002351, (3, 1): 0.0002651, (4, 1): 0.0002975, (5, 1): 0.0003327,
        (0, 2): 4.343e-05, (1, 2): 4.96e-05, (2, 2): 5.624e-05, (3, 2): 6.34e-05, (4, 2): 7.115e-05, (5, 2): 7.956e-05,
        (0, 3): 2.042e-05, (1, 3): 2.331e-05, (2, 3): 2.642e-05, (3, 3): 2.978e-05, (4, 3): 3.342e-05, (5, 3): 3.736e-05,
        (0, 4): 1.205e-05, (1, 4): 1.375e-05, (2, 4): 1.558e-05, (3, 4): 1.756e-05, (4, 4): 1.97e-05, (5, 4): 2.202e-05,
        (0, 5): 0.8022e-5
    }
    if in_au:
        factor = 1476.87
    else:
        factor = 1

    if (nu, L) in data:
        return data[(nu, L)]*factor
    else:
        print(f"E14({nu},{L}) not available.")
        return None

#The effective Hamiltonian contribution from the quadrupole coupling term, the gradient is given by Q (in V/m^2)
def VQ(nu,L,Q):
    states = states_FJM(L)
    N = len(states)
    Vmat = np.zeros((N,N))
    E_14 = np.sqrt(3/2)*E14(nu,L)
    if np.shape(Q) != (3,3):
        print("Q must be a 3x3 matrix.")
        return
    def Qhat(q):
        return 3/2*np.einsum('ij,ij->',Q,c_2(q))
    
    for i in range(N):
        F,J,MJ = states[i]
        J_= float(J)
        for q in [-2,1,0,1,2]:
            LxL2 = (-1)**(J+L+F)*w6j(L,2,L,J,F,J)*np.sqrt(2*J_+1) *np.sqrt(L*(L+1)*(2*L+1)*(2*L-1)*(2*L+3)/6)
            Vmat[i,i] += E_14*Qhat(q)*LxL2
    return Vmat


#Assuming electric field gradient only in zz direction
def VQ_only_Qzz(nu,L,Qzz):
    states = states_FJM(L)
    N = len(states)
    Vmat = np.zeros((N,N))
    E_14 = np.sqrt(3/2)*E14(nu,L)
    for i in range(N):
        F,J,MJ = states[i]
        J_= float(J)
        Vmat[i,i] = E_14*Qzz *(-1)**(J+L+F)*w6j(L,2,L,J,F,J)*np.sqrt(2*J_+1) *np.sqrt(L*(L+1)*(2*L+1)*(2*L-1)*(2*L+3)/6)
    return Vmat

def H_quadrupole(nu,L,B,Q):
    Heff = Htot(nu,L,B)
    if np.shape(Q) == (3,3):
        VQ = VQ_only_Qzz(nu,L,Q[2,2])
    elif np.shape(Q) == (1):
        VQ = VQ_only_Qzz(nu,L,Q)
    return Heff + VQ

########################################################################################################################
#The reduced matrix element <nu_g,Lg||Theta^(k)||nu_e,Le>, numerical results
def redTheta(nu_g,Lg,nu_e,Le):
    nu_L_g = nu_g, Lg
    nu_L_e = nu_e, Le

    #from Laser-stimulated electric quadrupole transitions V.I. Korobov (2018)
    Quadrupole_2018 = {
    ((0, 0), (0, 2)): 1.644960,
    ((0, 0), (1, 2)): -0.313846,
    ((0, 0), (2, 2)): -0.028919,
    ((0, 0), (3, 2)): -0.004652,
    ((0, 0), (4, 2)): -0.001048,
    ((0, 0), (5, 2)): -0.000298,
    ((0, 0), (6, 2)): -0.000100,
    ((0, 1), (0, 3)): 2.217064,
    ((0, 1), (1, 1)): 0.376163,
    ((0, 1), (1, 3)): -0.395911,
    ((0, 1), (2, 1)): 0.028875,
    ((0, 1), (2, 3)): -0.040697,
    ((0, 1), (3, 1)): 0.004044,
    ((0, 1), (3, 3)): -0.007056,
    ((0, 1), (4, 1)): 0.000792,
    ((0, 1), (4, 3)): -0.001698,
    ((0, 1), (5, 1)): 0.000192,
    ((0, 1), (5, 3)): 0.000514,
    ((0, 1), (6, 1)): -0.000053,
    ((0, 1), (6, 3)): -0.000185,
    ((0, 2), (0, 4)): 2.668039,
    ((0, 2), (1, 0)): -0.373540,
    ((0, 2), (1, 2)): 0.411812,
    ((0, 2), (1, 4)): -0.443985,
    ((0, 2), (2, 0)): -0.023126,
    ((0, 2), (2, 2)): 0.031686,
    ((0, 2), (2, 4)): -0.050500,
    ((0, 2), (3, 0)): -0.002661,
    ((0, 2), (3, 2)): 0.004448,
    ((0, 2), (3, 4)): 0.009347,
    ((0, 2), (4, 0)): -0.000398,
    ((0, 2), (4, 2)): 0.000874,
    ((0, 2), (4, 4)): 0.002372,
    ((0, 2), (5, 0)): -0.000058,
    ((0, 2), (5, 2)): 0.000213,
    ((0, 2), (5, 4)): -0.000753,
    ((0, 2), (6, 2)): 0.000059,
    ((0, 2), (6, 4)): -0.000282,
    ((0, 3), (0, 5)): 3.065234,
    ((0, 3), (1, 1)): -0.529496,
    ((0, 3), (1, 3)): 0.473113,
    ((0, 3), (1, 5)): 0.473145,
    ((0, 3), (2, 1)): -0.027758,
    ((0, 3), (2, 3)): 0.036532,
    ((0, 3), (2, 5)): -0.059153,
    ((0, 3), (3, 1)): -0.002632,
    ((0, 3), (3, 3)): 0.005146,
    ((0, 3), (3, 5)): 0.011606,
    ((0, 3), (4, 1)): -0.000253,
    ((0, 3), (4, 3)): 0.001015,
    ((0, 3), (4, 5)): 0.003080,
    ((0, 4), (2, 2)): -0.028888,
    ((1, 1), (1, 3)): 2.529827,
    ((1, 1), (3, 3)): -0.074474
    }

    if abs(Lg-Le) not in [0,2]:
        return 0
    elif (nu_L_g, nu_L_e) in Quadrupole_2018:
        return Quadrupole_2018[(nu_L_g, nu_L_e)]
    else:
        print(f"redTheta({nu_L_g},{nu_L_e}) not available.")
        return None

########################################################################################################################
#Calculations for quadrupole transitions

#The matrix (<(nu_g,Lg)Jg,Mg|Theta^(2)|(nu_e,Le)Je,Me>) for a quadrupole transition
def mat_Theta(q,nu_g,Lg,nu_e,Le):
    states_g = states_FJM(Lg)
    if Lg%2 ==0:
        Ig = 0
    else:
        Ig = 1
    states_e = states_FJM(Le)
    if Le%2 ==0:
        Ie = 0
    else:
        Ie = 1
    Ng = len(states_g)
    Ne = len(states_e)
    matrix_Theta = np.zeros((Ng,Ne))
    for i in range(Ng):
        for f in range(Ne):
            #get the quantum numbers of the states i and f
            Fg,Jg,MJg = states_g[i]
            Fe,Je,MJe = states_e[f]
            Fg_,Jg_,Je_,MJg_=float(Fg),float(Jg),float(Je),float(MJg)
            if Ig == Ie and Fg == Fe:
                redTheta_Js = (-1)**(Je_+Lg+Fg_) *np.sqrt((2*Je_+1)) *w6j(Lg,2,Le,Je,Fg,Jg) *redTheta(nu_g,Lg,nu_e,Le)
                matrix_Theta[i,f] = CG(Je,MJe,2,q,Jg,MJg).doit() *redTheta_Js
    return matrix_Theta

def mixed_mat_Theta(q,nu_g,Lg,nu_e,Le,B):
    _,ESs_g = get_EWs_EVs_B0(nu_g,Lg,B)
    _,ESs_e = get_EWs_EVs_B0(nu_e,Le,B)
    matrix_Theta = mat_Theta(q,nu_g,Lg,nu_e,Le)
    return np.einsum('ik,ij,jl->kl',ESs_g,matrix_Theta,ESs_e)

#epsilon is the polarization of the electric field, n is the unit vector along the propagation direction of the electric field
def quadrupole_operator_pure(nu_i,Li,nu_f,Lf,epsilon,n):
    states_i = states_FJM(Li)
    states_f = states_FJM(Lf)
    Ni = len(states_i)
    Nf = len(states_f)

    mat_quad = np.zeros((Ni,Nf), dtype=complex)
    for q in [-2,-1,0,1,2]:
        mat_Theta_q = mat_Theta(q,nu_i,Li,nu_f,Lf)
        g_q = np.einsum('i,ij,j->',epsilon,c_2(q),n) #geometric factor g_q
        mat_quad += 1/3 *mat_Theta_q *g_q
    return mat_quad

def quadrupole_operator_mixed(nu_i,Li,EVs_i,nu_f,Lf,EVs_f,epsilon,n):

    Qop_pure = quadrupole_operator_pure(nu_i,Li,nu_f,Lf,epsilon,n)
    return np.einsum('ik,ij,jl->kl',EVs_i,Qop_pure,EVs_f)

def Thetaif_en(nu_i,Li,nu_f,Lf):
    dL = abs(Li - Lf)
    #dMJ = MJ_i - MJ_f
    if dL == 0:
        n = n_0
        eps = eps_0
    elif dL == 2:
        n = n_pm2
        eps = eps_pm2
    else:
        return 0
    
    if Li%2 == 1:
        F_i,J_i,MJ_i = 3/2, Li + 3/2, +(Li + 3/2)
        F_f,J_f,MJ_f = 3/2, Lf + 3/2, +(Lf + 3/2)
    else: #Even Ls: not as in the paper, not sure which state they use
        F_i,J_i,MJ_i = 1/2, Li + 1/2, +(Li + 1/2)
        F_f,J_f,MJ_f = 1/2, Lf + 1/2, +(Lf + 1/2)
    
    index_i = states_FJM_index(F_i,J_i,MJ_i,Li)
    index_f = states_FJM_index(F_f,J_f,MJ_f,Lf)
    #quadrupole_operator_pure or quadrupole_operator_mixed should return the same result for these streched (pure) states
    return quadrupole_operator_pure(nu_i,Li,nu_f,Lf,eps,n)[index_i,index_f]

########################################################################################################################
#Polarizabilities

#Static dipole polarizability tensor in au.
#Scalar term
def alpha_s(nu,L,rel_corr=True):
    #from Table II in V. I. Korobov (2016)
    data_2016 = {
        (0, 0): 3.1685731, (0, 1): 3.1781425, (0, 2): 3.1973545,
        (0, 3): 3.2262879, (0, 4): 3.2650990, (0, 5): 3.3139976,
        (1, 0): 3.8973934, (1, 1): 3.9099178, (1, 2): 3.9350819,
        (1, 3): 3.9730164, (1, 4): 4.0239695, (1, 5): 4.0882763,
        (2, 0): 4.8213113, (2, 1): 4.8378793, (2, 2): 4.8711902,
        (2, 3): 4.9214594, (2, 4): 4.9890778, (2, 5): 5.0745756,
        (3, 0): 6.0091177, (3, 1): 6.0313112, (3, 2): 6.0759600,
        (3, 3): 6.1434165, (3, 4): 6.2342968, (3, 5): 6.3494400,
        (4, 0): 7.5602216, (4, 1): 7.5903867, (4, 2): 7.6511105,
        (4, 3): 7.7429690, (4, 4): 7.8669387, (4, 5): 8.0243574,
        (5, 0): 9.6215210, (5, 1): 9.6632217, (5, 2): 9.7472225,
        (5, 3): 9.8744707, (5, 4): 10.046534, (5, 5): 10.265571
    }
    #from Table II in Schiller et al., PRA 89, 052521 (2014)
    data_2014 = {
        (0, 0): 3.1687258, (0, 1): 3.1783035, (0, 2): 3.1975081,
        (0, 3): 3.2264392, (0, 4): 3.2652493, (0, 5): 3.3141473,
        (1, 0): 3.8975634, (1, 1): 3.9101018, (1, 2): 3.9352574,
        (1, 3): 3.9731892, (1, 4): 4.0241411, (1, 5): 4.0884471,
        (2, 0): 4.8215004, (2, 1): 4.8380889, (2, 2): 4.8713900,
        (2, 3): 4.9216560, (2, 4): 4.9892726, (2, 5): 5.0747693,
        (3, 0): 6.0093275, (3, 1): 6.0315483, (3, 2): 6.0761862,
        (3, 3): 6.1436389, (3, 4): 6.2345166, (3, 5): 6.3496578,
        (4, 0): 7.5604532, (4, 1): 7.5906530, (4, 2): 7.6513642,
        (4, 3): 7.7432180, (4, 4): 7.8671844, (4, 5): 8.0246002,
        (5, 0): 9.6217735, (5, 1): 9.6635170, (5, 2): 9.7475033,
        (5, 3): 9.8747452, (5, 4): 10.046804, (5, 5): 10.265837
    }

    if (nu, L) in data_2016:
        if rel_corr:
            return data_2016[(nu, L)]
        else:
            return data_2014[(nu, L)]
    
    else:
        print(f"αs({nu},{L}) not available.")
        return None
#Tensor term
def alpha_t(nu,L,rel_corr=True):
    #from Table II in V. I. Korobov (2016)
    data_2016 = {
        (0, 1): -0.8033502, (0, 2): -0.1931356, (0, 3): -0.0914433,
        (0, 4): -0.0544748, (0, 5): -0.0367128,
        (1, 1): -1.1441799, (1, 2): -0.2750942, (1, 3): -0.1302617,
        (1, 4): -0.0776116, (1, 5): -0.0523165,
        (2, 1): -1.6000406, (2, 2): -0.3847577, (2, 3): -0.1822335,
        (2, 4): -0.1086134, (2, 5): -0.0732459,
        (3, 1): -2.2129254, (3, 2): -0.5322677, (3, 3): -0.2521933,
        (3, 4): -0.1503867, (3, 5): -0.1014829,
        (4, 1): -3.0434518, (4, 2): -0.7322788, (4, 3): -0.3471380,
        (4, 4): -0.2071473, (4, 5): -0.1399094,
        (5, 1): -4.1811193, (5, 2): -1.0064538, (5, 3): -0.4774294,
        (5, 4): -0.2851531, (5, 5): -0.1928167
    }

    #from Table II in Schiller et al., PRA 89, 052521 (2014)
    data_2014 = {
        (0, 1): -0.8033729, (0, 2): -0.1931423, (0, 3): -0.0914467,
        (0, 4): -0.0544769, (0, 5): -0.0367142,
        (1, 1): -1.1442051, (1, 2): -0.2751013, (1, 3): -0.1302653,
        (1, 4): -0.0776138, (1, 5): -0.0523179,
        (2, 1): -1.6000689, (2, 2): -0.3847653, (2, 3): -0.1822373,
        (2, 4): -0.1086157, (2, 5): -0.0732474,
        (3, 1): -2.2129563, (3, 2): -0.5322759, (3, 3): -0.2521973,
        (3, 4): -0.1503892, (3, 5): -0.1014845,
        (4, 1): -3.0434869, (4, 2): -0.7322875, (4, 3): -0.3471422,
        (4, 4): -0.2071498, (4, 5): -0.1399110,
        (5, 1): -4.1811566, (5, 2): -1.0064626, (5, 3): -0.4774336,
        (5, 4): -0.2851555, (5, 5): -0.1928182
    }

    if (nu, L) in data_2016:
        if rel_corr:
            return data_2016[(nu, L)]
        data_2014[(nu, L)]
    else:
        print(f"αt({nu},{L}) not available.")
        return None

#Computes the polarizabilities of the pure hyperfine states
#for a static electric field along the quantization axis (alpha_para) and perpendicular to it (alpha_ortho)
def alpha_para_ortho_pure(nu,L):
    states = states_FJM(L)
    N = len(states)
    LtensorL_20 = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            F1,J1,M1 = states[i]
            F2,J2,M2 = states[j]
            
            J1_,J2_ = float(J1), float(J2)

            if F1==F2:
                LtensorL_20[i,j] = CG(J2,M2,2,0,J1,M1).doit() *(-1)**(J2+L+F1) *w6j(L,2,L,J2,F1,J1) *np.sqrt(2*J2_+1) \
                                    *1/np.sqrt(6) *np.sqrt(L*(L+1)*(2*L-1)*(2*L+1)*(2*L+3))
    if np.all(LtensorL_20 == 0):
        a_pa = alpha_s(nu,L)*np.eye(N)
        a_or = alpha_s(nu,L)*np.eye(N)
    else:
        a_pa = alpha_s(nu,L)*np.eye(N) + alpha_t(nu,L)*np.sqrt(8/3)*LtensorL_20
        a_or = alpha_s(nu,L)*np.eye(N) - 1/2 *alpha_t(nu,L)*np.sqrt(8/3)*LtensorL_20

    return a_pa, a_or

#Computes the polarizabilities for the mixed states at a given magnetic field B
def alphas_mixed_B(nu,L,B, printit=False):
    EWs,EVs = get_EWs_EVs_B0(nu,L,B)
    N = len(EWs)
    
    ap_pure, ao_pure = alpha_para_ortho_pure(nu,L)
    
    ap, ao = np.einsum('km,kl,ln->mn',EVs,ap_pure,EVs), np.einsum('km,kl,ln->mn',EVs,ao_pure,EVs)
    
    # Your precision value (adjust as needed)
    prec = 12

    # Set the printing options
    np.set_printoptions(suppress=True, precision=prec)
    if printit:
        print(ap,ao)
    
    return ap, ao

#Computes polarizability for a static E-Field at an angle theta to the quantization axis
def alpha_theta(nu,L,B,theta,mixed=False):
    if mixed:
        ap,ao = alphas_mixed_B(nu,L,B)
    else:
        ap,ao = alpha_para_ortho_pure(nu,L)
    return ap*np.cos(theta)**2 + ao*np.sin(theta)**2

########################################################################################################################
#Frequency shifts

#DC Stark shift
def DC_Stark_shift(nu,L,B, Efield):
    states = states_FJM(L)
    N = len(states)
    alpha_para, alpha_orth = alphas_mixed_B(nu,L,B)
    Stark_shift = 0.5*(np.diag(alpha_para)*Efield[0]**2 + np.diag(alpha_orth)*(Efield[1]**2+Efield[2]**2))
    return Stark_shift

#AC Stark shift
def AC_Stark_shift_quadrupole_transition(nu_i,Li,i,nu_f,Lf,f,B,epsilon,n,pulsetime):
    EWs_i,EVs_i = get_EWs_EVs_B0(nu_i,Li,B)
    EWs_f,EVs_f = get_EWs_EVs_B0(nu_f,Lf,B)

    #approximate polarizabilities by the value at w=0
    ap_i,ao_i = alphas_mixed_B(nu_i,Li,B)
    ap_f,ao_f = alphas_mixed_B(nu_f,Lf,B)
    #get angle between quantization axis and electric field (cos^2 and sin^2 of the angle)
    costheta2,sintheta2 = n[2]**2, n[0]**2+n[1]**2
    #calculate polarizabilities for the electric field at an angle theta to the quantization axis
    a_i = costheta2 *ap_i[i,i] + sintheta2 *ao_i[i,i]
    a_f = costheta2 *ap_f[f,f] + sintheta2 *ao_f[f,f]
    delta_alpha = (a_f - a_i) #*4*pi*epsilon_0*a_0**3 #Delta_alpha in SI units

    Quadop = quadrupole_operator_mixed(nu_i,Li,EVs_i,nu_f,Lf,EVs_f,epsilon,n)
    Thetaif = np.einsum('km,kl,ln->mn',EVs_i,Quadop,EVs_f)[i,f]

    if Thetaif == 0:
        print(f"Quadrupole transition from {nu_i,Li} state {i} to {nu_f,Lf} state {f} is forbidden for ε={epsilon}, n={n}.")
        return

    #Frequencies in MHz
    freq_NR = Energy_diff(nu_i, Li, nu_f, Lf, rel=False)
    freq = (EWs_f[f]-EWs_i[i])
    #Wavelenght in m
    transition_wavelenght = c/(freq+freq_NR)*1e-6

    return -m_e *transition_wavelenght**2 /(4*hbar *pulsetime**2) *delta_alpha/np.abs(Thetaif)**2

def AC_Stark_shift_quadrupole_levels(nu_i,Li,nu_f,Lf,B,epsilon,n,pulsetime,mixed=True):
    EWs_i,EVs_i = get_EWs_EVs_B0(nu_i,Li,B)
    EWs_f,EVs_f = get_EWs_EVs_B0(nu_f,Lf,B)

    #approximate polarizabilities by the value at w=0
    if mixed:
        ap_i,ao_i = alphas_mixed_B(nu_i,Li,B)
        ap_f,ao_f = alphas_mixed_B(nu_f,Lf,B)
    else:
        ap_i,ao_i = alpha_para_ortho_pure(nu_i,Li)
        ap_f,ao_f = alpha_para_ortho_pure(nu_f,Lf)
    #get angle between quantization axis and electric field (cos^2 and sin^2 of the angle)
    costheta2,sintheta2 = n[2]**2, n[0]**2+n[1]**2
    #calculate polarizabilities for the electric field at an angle theta to the quantization axis
    a_i = np.diag(costheta2 *ap_i + sintheta2 *ao_i)
    a_f = np.diag(costheta2 *ap_f + sintheta2 *ao_f)

    if mixed:
        Quadop = quadrupole_operator_mixed(nu_i,Li,EVs_i,nu_f,Lf,EVs_f,epsilon,n)
        Theta = np.einsum('km,kl,ln->mn',EVs_i,Quadop,EVs_f)
    else:
        Theta = quadrupole_operator_pure(nu_i,Li,nu_f,Lf,epsilon,n)
    
    #Frequencies in MHz
    freq_NR = Energy_diff(nu_i, Li, nu_f, Lf, rel=False)
    freqs = mat_transition_freqs_E(nu_i, Li, EWs_i, nu_f, Lf, EWs_f)

    shift = np.zeros((len(EWs_i),len(EWs_f)))
    for i in range(len(EWs_i)):
        for f in range(len(EWs_f)):
            delta_alpha = (a_i[i] - a_f[f]) #*4*pi*epsilon_0*a_0**3 #Delta_alpha in SI units
            Thetaif = abs(Theta[i,f])

            transition_wavelenght = c/(freqs[i,f]+freq_NR)*1e-6
            if Thetaif >= 1e-12:
                shift[i,f] = -m_e *transition_wavelenght**2 /(4*hbar *pulsetime**2) *delta_alpha/np.abs(Thetaif)**2
                if shift[i,f] == 0:
                    print(f"Quadrupole transition from {nu_i,Li} state {i} to {nu_f,Lf} state {f} has shift 0 for ε={epsilon}, n={n}.")
            else:
                shift[i,f] = 0
    return shift

#assume angle theta between quantization axis and electric field (default =0)
def AC_Stark_shift_twophoton_transition(nu_i,Li,i,nu_f,Lf,f,B,q1,q2,pulsetime,theta=0):
    EWs_i,EVs_i = get_EWs_EVs_B0(nu_i,Li,B)
    EWs_f,EVs_f = get_EWs_EVs_B0(nu_f,Lf,B)


    #approximate polarizabilities by the value at w=0
    ap_i,ao_i = alphas_mixed_B(nu_i,Li,B)
    ap_f,ao_f = alphas_mixed_B(nu_f,Lf,B)
    #get angle between quantization axis and electric field (cos^2 and sin^2 of the angle)
    costheta2,sintheta2 = np.cos(theta)**2, np.sin(theta)**2
    #calculate polarizabilities for the electric field at an angle theta to the quantization axis
    a_i = np.diag(costheta2 *ap_i + sintheta2 *ao_i)
    a_f = np.diag(costheta2 *ap_f + sintheta2 *ao_f)
    delta_alpha = a_i[i,i] - a_f[f,f]

    SQif = mat_SQ_mix(q1,q2,nu_i,Li,EVs_i,nu_f,Lf,EVs_f)[i,f]

    if SQif == 0:
        print(f"2 photon transition from {nu_i,Li} state {i} to {nu_f,Lf} state {f} is forbidden for q1={q1}, q2={q2}.")
        return

    return -1/(8*pulsetime) *delta_alpha/np.abs(SQif)

def AC_Stark_shift_twophoton_levels(nu_i,Li,nu_f,Lf,B,q1,q2,pulsetime,theta=0):
    EWs_i,EVs_i = get_EWs_EVs_B0(nu_i,Li,B)
    EWs_f,EVs_f = get_EWs_EVs_B0(nu_f,Lf,B)

    #approximate polarizabilities by the value at w=0
    ap_i,ao_i = alphas_mixed_B(nu_i,Li,B)
    ap_f,ao_f = alphas_mixed_B(nu_f,Lf,B)
    #get angle between quantization axis and electric field (cos^2 and sin^2 of the angle)
    costheta2,sintheta2 = np.cos(theta)**2, np.sin(theta)**2
    #calculate polarizabilities for the electric field at an angle theta to the quantization axis
    a_i = np.diag(costheta2 *ap_i + sintheta2 *ao_i)
    a_f = np.diag(costheta2 *ap_f + sintheta2 *ao_f)

    SQ = mat_SQ_mix(q1,q2,nu_i,Li,EVs_i,nu_f,Lf,EVs_f)

    shift = np.zeros((len(EWs_i),len(EWs_f)))
    for i in range(len(EWs_i)):
        for f in range(len(EWs_f)):
            delta_alpha = (a_f[f] - a_i[i]) #*4*pi*epsilon_0*a_0**3 #Delta_alpha in SI units
            SQif = abs(SQ[i,f])
            if SQif >= 1e-12:
                shift[i,f] = -1/(8*pulsetime) *delta_alpha/SQif
                if shift[i,f] == 0:
                    print(f"2 photon transition from {nu_i,Li} state {i} to {nu_f,Lf} state {f} has shift 0 for q1={q1}, q2={q2}.")
            else:
                shift[i,f] = 0
    return shift

#BBR shift for a given rovibrational state
def BBR_shift_stat(nu,L,T):
    a_s = alpha_s(nu,L)*4*pi*epsilon_0*a_0**3 #alpha_s in SI units
    return -1/2 *a_s *831.9**2 *(T/300)**4

########################################################################################################################
#Dipole moment matrix elements for mixed states at a given magnetic field B
def mu_mixed(q, nu, L, EVs=None, B=None, accuracy=1e-12):
    """
    This function calculates the dipole moment matrix elements for the mixed states at a given magnetic field B.

    Parameters
    ----------
    q : int
        The polarization of the photon.
    nu : float
        The vibrational quantum number of the state.
    L : int
        The total orbital angular momentum quantum number of the state.
    EVs : ndarray, optional
        Array containing all the c_(F,J,MJ) coefficients of the eigenstates.
    B : float, optional
        The magnetic field strength.

    Returns
    -------
    matrix : ndarray
        The matrix of dipole moment matrix elements for the mixed states at a given magnetic field B.
    """
    if EVs is None:
        if B is None:
            raise ValueError("If EVs is not provided, B must be provided to calculate EVs.")
        EWs, EVs = get_EWs_EVs_B0(nu, L, B)

    N = len(EVs)
    matrix = np.zeros((N, N))
    
    mu_pure = mu(q, nu, L)
    
    #for i in range(N):
    #    for f in range(N):
    #        F_i, J_i, MJ_i = states_FJM(L)[i]
    #        F_f, J_f, MJ_f = states_FJM(L)[f]
    #        if F_i == F_f:
    #            matrix[i, f] = sum(EVs[k, i] * EVs[l, f] * mu_pure[k,l] for k in range(N) for l in range(N))
    
    # Calculate the matrix product
    matrix = np.dot(EVs.T, np.dot(mu_pure, EVs))

    # Set all values less than the accuracy to 0
    matrix[np.abs(matrix) < accuracy] = 0
    
    return matrix

def mu_div_mub_mixed(q, nu, L, EVs=None, B=None, accuracy=1e-12):
    """
    This function calculates the dipole moment matrix elements for the mixed states at a given magnetic field B.

    Parameters
    ----------
    q : int
        The polarization of the photon.
    nu : float
        The vibrational quantum number of the state.
    L : int
        The total orbital angular momentum quantum number of the state.
    EVs : ndarray, optional
        Array containing all the c_(F,J,MJ) coefficients of the eigenstates.
    B : float, optional
        The magnetic field strength.

    Returns
    -------
    matrix : ndarray
        The matrix of dipole moment matrix elements for the mixed states at a given magnetic field B.
    """
    if EVs is None:
        if B is None:
            raise ValueError("If EVs is not provided, B must be provided to calculate EVs.")
        EWs, EVs = get_EWs_EVs_B0(nu, L, B)

    N = len(EVs)
    matrix = np.zeros((N, N))
    
    mu_pure = mu_div_mub(q, nu, L)
    
    # Calculate the matrix product
    matrix = np.dot(EVs.T, np.dot(mu_pure, EVs))

    # Set all values less than the accuracy to 0
    matrix[np.abs(matrix) < accuracy] = 0
    
    return matrix

########################################################################################################################
########################################################################################################################
#Functions for transitions
import pandas as pd
from ipywidgets import interact, interactive, Dropdown, FloatLogSlider, FloatSlider, IntText, Checkbox

#sets a color for each different set of states (F,J) that is displayed in the plots given the L level
def lev_col(L,F,J):
    if F==0.5 and J==L-0.5:
        return 'b'
    elif F==0.5 and J==L+0.5:
        return 'r'
    elif F==1.5 and J==L-1.5:
        return 'g'
    elif F==1.5 and J==L-0.5:
        return 'y'
    elif F==1.5 and J==L+0.5:
        return 'm'
    elif F==1.5 and J==L+1.5:
        return 'c'
    else:
        raise ValueError("Invalid values of F and J.")


#Find possible magnetic field insensitive transitions
def insensitive_transitions_same(nu,L,Bvalues,threshold=1):
    states = states_FJM(L)
    N = len(states)
    
    n_B = len(Bvalues)
    
    _,_,derivatives = get_EWs_EVs_derivatives(nu, L, Bvalues)
    min_sensibilities = {}
    for k in range(n_B):
        for i in range(N):
            for f in range(i+1,N):
                sensibility_if = derivatives[f,k]-derivatives[i,k]
                if abs(sensibility_if) < threshold:
                    transition = (i,f)
                    if transition not in min_sensibilities or abs(sensibility_if) < abs(min_sensibilities[transition][3]):
                        min_sensibilities[transition] = (Bvalues[k],i, f, sensibility_if)
                        #print(min_sensibilities[transition][3])
    sensibilities = list(min_sensibilities.values())
    return sensibilities

def insensitive_transitions(nu_i,Li,nu_f,Lf,Bvalues,threshold=1):
    if nu_i==nu_f and Li==Lf:
        return insensitive_transitions_same(nu_i,Li,Bvalues,threshold)
    states_i = states_FJM(Li)
    states_f = states_FJM(Lf)
    Ni = len(states_i)
    Nf = len(states_f)
    n_B = len(Bvalues)
    
    _,_,derivatives_i = get_EWs_EVs_derivatives(nu_i, Li, Bvalues)
    _,_,derivatives_f = get_EWs_EVs_derivatives(nu_f, Lf, Bvalues)
    min_sensibilities = {}
    for k in range(n_B):
        for i in range(Ni):
            for f in range(Nf):
                sensibility_if = derivatives_f[f,k]-derivatives_i[i,k]
                if abs(sensibility_if) < threshold:
                    transition = (i,f)
                    if transition not in min_sensibilities or abs(sensibility_if) < abs(min_sensibilities[transition][3]):
                        min_sensibilities[transition] = (Bvalues[k],i, f, sensibility_if)
                        #print(min_sensibilities[transition][3])
    sensibilities = list(min_sensibilities.values())
    return sensibilities

#This is the best one
def insensitive_transitions_ranges(nu_i,Li,derivatives_i,nu_f,Lf,derivatives_f,Bvalues,threshold=1):
    states_i = states_FJM(Li)
    states_f = states_FJM(Lf)
    Ni = len(states_i)
    Nf = len(states_f)
    n_B = len(Bvalues)

    ranges_sensibilities = []
    Bmins = np.zeros((Ni,Nf))
    insensitive = np.zeros((Ni,Nf))
    min_sens = np.zeros((Ni,Nf))
    Bsens = np.zeros((Ni,Nf))
    
    for k in range(n_B):
        for i in range(Ni):
            #For the same initial and final level, we only need to consider the upper triangle of the matrix
            if nu_i==nu_f and Li==Lf:
                range_f = range(i+1,Nf)
            else:
                range_f = range(Nf)
            
            for f in range_f:
                sensibility_if = derivatives_f[f,k]-derivatives_i[i,k]
                if insensitive[i,f]==0 and abs(sensibility_if) < threshold:
                    Bmins[i,f] = Bvalues[k]
                    insensitive[i,f] = 1
                    min_sens[i,f] = sensibility_if
                    Bsens[i,f] = Bvalues[k]
                    index_min = k
                elif insensitive[i,f]==1 and abs(sensibility_if) < abs(min_sens[i,f]):
                    min_sens[i,f] = sensibility_if
                    Bsens[i,f] = Bvalues[k]
                    index_min = k
                elif insensitive[i,f]==1 and abs(sensibility_if) > threshold:
                    Bmax = Bvalues[k-1]
                    Brange = np.array([Bmins[i,f],Bmax])
                    ranges_sensibilities.append((i,f,Brange,min_sens[i,f],Bsens[i,f],index_min))
                    insensitive[i,f] = 0
                elif insensitive[i,f]==1 and k == n_B-1:
                    Bmax = Bvalues[k]
                    Brange = np.array([Bmins[i,f],Bmax])
                    ranges_sensibilities.append((i,f,Brange,min_sens[i,f],Bsens[i,f], index_min))
                    insensitive[i,f] = 0
    return ranges_sensibilities

def insensitive_transition(nu_i,Li,i,nu_f,Lf,f,Bvalues,threshold=1):
    states_i = states_FJM(Li)
    states_f = states_FJM(Lf)
    Ni = len(states_i)
    Nf = len(states_f)
    n_B = len(Bvalues)
    
    _,_,derivatives_i = get_EWs_EVs_derivatives(nu_i, Li, Bvalues)
    _,_,derivatives_f = get_EWs_EVs_derivatives(nu_f, Lf, Bvalues)

    insensitive_range = []
    min_sensibilities = []
    Bmin = 0
    insensitive = False
    for k in range(n_B):
        sensibility_if = derivatives_f[f,k]-derivatives_i[i,k]
        if not insensitive and abs(sensibility_if) < threshold:
            Bmin = Bvalues[k]
            min_sens = np.array((Bvalues[k],sensibility_if))
            insensitive = True
        elif insensitive and abs(sensibility_if) < threshold:
            if abs(sensibility_if) < abs(min_sens[1]):
                min_sens = np.array([Bvalues[k],sensibility_if])
        elif insensitive and abs(sensibility_if) > threshold:
            Bmax = Bvalues[k-1]
            Brange = np.array([Bmin,Bmax])
            insensitive_range.append(Brange)
            min_sensibilities.append(min_sens)
            insensitive = False
        elif insensitive and k == n_B-1:
            Bmax = Bvalues[k]
            Brange = np.array([Bmin,Bmax])
            insensitive_range.append(Brange)
            min_sensibilities.append(min_sens)
            
    return i,f,np.array(insensitive_range),np.array(min_sensibilities)

############################################################################################################
#Table of hyperfine transitions for a given (nu,L) level of H2+ and a given magnetic field value B
def table_hf_transitions(nu,L,B,prob_threshold=1e-3, sensitivity_threshold=None):
    """
    Table of hyperfine transitions for a given (nu,L) level of H2+ and a given magnetic field value B.
    This function calculates the energy difference, magnetic field sensitivity, and transition probabilities for all hyperfine transitions for a given (nu,L) level of H2+ and a given magnetic field value B.
    It only shows transitions that have a probability above a certain threshold.

    Parameters
    ----------
    nu : int
        The nu level.
    L : int
        The L level.
    B : float
        The magnetic field value.
    prob_threshold : float, optional
        The probability threshold for showing transitions. Default is None, which shows all transitions.
    sensitivity_threshold : float, optional
        The sensitivity threshold for showing transitions. Default is None, which shows all transitions.

    Returns
    -------
    DataFrame
        The table of hyperfine transitions.
    """
    # Get the eigenvalues, eigenvectors and derivatives for the given B
    EWs_B, EVs_B,dE_B_ = get_EWs_EVs_derivatives_B0(nu, L, B)

    # Get the number of states
    N = len(EWs_B)
    
    
    mu_p = mu_div_mub_mixed(1, nu, L,EVs=EVs_B)
    mu_0 = mu_div_mub_mixed(0, nu, L,EVs=EVs_B)
    mu_m = mu_div_mub_mixed(-1, nu, L,EVs=EVs_B)

    # Initialize empty lists to store the transition data
    transitions = []
    for i in range(N):
        for j in range(i+1,N):
            # Calculate the energy difference
            diff_Eij = EWs_B[j]-EWs_B[i]
            # Calculate the magnetic field sensitivity
            diff_dE_Bij = dE_B_[j]-dE_B_[i]
            
            # If the transition probability is above the threshold, add the transition data to the list
            if prob_threshold is None or abs(mu_p[i,j]) > prob_threshold or abs(mu_0[i,j]) > prob_threshold or abs(mu_m[i,j]) > prob_threshold:
                if sensitivity_threshold is None or abs(diff_dE_Bij) < sensitivity_threshold:
                    transitions.append([i, j, diff_Eij, diff_dE_Bij, mu_p[i,j], mu_0[i,j], mu_m[i,j]])
    # Create a DataFrame from the list of transitions
    df = pd.DataFrame(transitions, columns=['Initial State', 'Final State', 'Transition frequency [MHz]', \
                             'Magnetic Field Sensitivity [MHz/T]', '$\\mu_+ / \\mu_b$', '$\\mu_0 / \\mu_b$', '$\\mu_- / \\mu_b$'])
    return df

def table_insensitive_hf_transitions(nu,L,Bmin,Bmax,samples=1000, prob_threshold=1e-3, sensitivity_threshold=1):
    """
    Table of hyperfine transitions for a given (nu,L) level of H2+ and the magnetic field value at which the transitions are insensitive to the magnetic field.
    This function calculates the energy difference, magnetic field sensitivity, and transition probabilities for all hyperfine transitions for a given (nu,L) level of H2+ and the magnetic field value at which the transitions are insensitive to the magnetic field.

    Parameters
    ----------
    nu : int

    L : int

    Bmax : float
        The maximal magnetic field value to consider for the range of magnetic field values.

    prob_threshold : float, optional
        The probability threshold for showing transitions. Default is 1e-3.

    sensitivity_threshold : float, optional
        The sensitivity threshold for considering a transition insensitive. Default is 1 [MHz/T].

    Returns
    -------
    DataFrame
        The table of hyperfine transitions.
    """
    Bvalues = np.linspace(Bmin, Bmax, samples)
    B_values_zero_sensibility = insensitive_transitions_same(nu,L,Bvalues,threshold=sensitivity_threshold)

    insensitive_hf_transitions = []

    for B, i, f, sensibility_if in B_values_zero_sensibility:
        EWs_B, EVs_B = get_EWs_EVs_B0(nu, L, B)
        mu_p = mu_div_mub_mixed(1, nu, L,EVs=EVs_B)
        mu_0 = mu_div_mub_mixed(0, nu, L,EVs=EVs_B)
        mu_m = mu_div_mub_mixed(-1, nu, L,EVs=EVs_B)
        if prob_threshold is None or abs(mu_p[i,f]) > prob_threshold or abs(mu_0[i,f]) > prob_threshold or abs(mu_m[i,f]) > prob_threshold:
            insensitive_hf_transitions.append([B, i, f, EWs_B[f]-EWs_B[i], sensibility_if, mu_p[i,f], mu_0[i,f], mu_m[i,f]])
    if len(insensitive_hf_transitions) == 0:
        print('No insensitive transitions with the given thresholds.')
        return
    display(Markdown(f'The insensitive transitions of the rovibrational level $(\\nu,\\mathbf{{L}})=({nu},{L})$ for magnetic fields up to {Bmax} T are shown in the table below.'))

    df = pd.DataFrame(insensitive_hf_transitions, columns=['Magnetic Field [T]', 'Initial State', 'Final State', 'Transition frequency [MHz]', \
                             'Magnetic Field Sensitivity [MHz/T]', '$\\mu_+ / \\mu_b$', '$\\mu_0 / \\mu_b$', '$\\mu_- / \\mu_b$'])
    return df

############################################################################################################
#Plots the eigenenergies as a function of the magnetic field for a given (nu,L) level of H2+
def plot_eigenenergies(nu, L, B_max, statesFJ_indices=None, number_points_B=1000, separate=True, B_min=0, show_derivatives=True):
    #Get the eigenvalues, eigenvectors and derivatives
    B_values = np.linspace(B_min, B_max, number_points_B)
    eigenvalues_, eigenvectors_, derivatives_ = get_EWs_EVs_derivatives(nu, L, B_values)
    
    statesFJ = states_FJ(L)
    statesFJM = states_FJM(L)

    #Get the number of states
    Nred = len(statesFJ)
    N = len(statesFJM)


    ptit = 'Eigenenergies for $(\\nu,\\mathbf{{L}})$=({0},{1})'.format(nu,L)

    #rescale x-axis of the plot
    if B_max<=1e-3:
        Bplot = B_values*1e6
        xlab = 'B (µT)'
    elif B_max<=1:
        Bplot = B_values*1e3
        xlab = 'B (mT)'
    else:
        Bplot = B_values
        xlab = 'B (T)'

    # If statesFJ_indices is None, set it to be all indices
    if statesFJ_indices is None:
        statesFJ_indices = list(range(len(statesFJ)))
        Nstates = Nred
    else:
        Nstates = len(statesFJ_indices)
    # Get the states corresponding to the indices
    statesFJ_selected = [statesFJ[i] for i in statesFJ_indices]


    # Plot the eigenvalues as a function of B for the different F,J
    for i in range(Nstates):
        if separate or i == 0:  # Create a new figure for the first pair or if Separate is checked
            if show_derivatives:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            else:
                fig, ax1 = plt.subplots(figsize=(6, 6))

        F,J = statesFJ_selected[i]
        col = lev_col(L, F, J)
        label_shown = False
        for k in range(int(2*J+1)):
            MJ = k-J
            energy = eigenvalues_[states_FJM_index(F, J, MJ, L)]
            ax1.plot(Bplot, energy, color=col)
            if not label_shown:
                ax1.plot([],[],color=col, label='F={}, J={}'.format(F, J))
                label_shown = True

            # Plot the derivative of the energy if the Derivatives checkbox is checked
            if show_derivatives:
                derivative = derivatives_[states_FJM_index(F, J, MJ, L)]
                #plot derivative in MHz/mT for better readability
                ax2.plot(Bplot, derivative*1e-3, color=col, linestyle='--')

        ax1.set_title(ptit)
        ax1.set_xlabel(xlab)
        ax1.set_ylabel('Eigenenergies (MHz)')
        ax1.legend()

        # Set labels for the derivative plot
        if show_derivatives:
            ax2.set_title('Derivative of ' + ptit)
            ax2.set_xlabel(xlab)
            ax2.set_ylabel('Derivative of Eigenenergies (MHz/mT)')

        plt.tight_layout()
        if separate:  # Display the figure after each pair if Separate is checked
            plt.show()

    if not separate:  # Display the figure after all pairs if Separate is not checked
        plt.show()

def plot_derivatives(nu, L, B_max, statesFJ_indices=None, number_points_B=1000, separate=True, B_min=0):
    # Get the eigenvalues, eigenvectors and derivatives
    B_values = np.linspace(B_min, B_max, number_points_B)
    _,_,derivatives_ = get_EWs_EVs_derivatives(nu, L, B_values)
    
    statesFJ = states_FJ(L)
    statesFJM = states_FJM(L)

    # If statesFJ_indices is None, set it to be all indices
    if statesFJ_indices is None:
        statesFJ_indices = list(range(len(statesFJ)))
    # Get the states corresponding to the indices
    statesFJ_selected = [statesFJ[i] for i in statesFJ_indices]

    ptit = 'Derivative of Eigenenergies for $(\\nu,\\mathbf{{L}})$=({0},{1})'.format(nu,L)

    # Rescale x-axis of the plot
    B_range = B_max-B_min
    if B_range<=1e-3:
        Bplot = B_values*1e6
        xlab = 'B (µT)'
    elif B_range<=1:
        Bplot = B_values*1e3
        xlab = 'B (mT)'
    else:
        Bplot = B_values
        xlab = 'B (T)'

    # Plot the derivatives for the different F,J
    for i in range(len(statesFJ_indices)):
        if separate or i == 0:  # Create a new figure for the first pair or if Separate is checked
            fig, ax = plt.subplots(figsize=(6, 6))

        F,J = statesFJ_selected[i]
        col = lev_col(L, F, J)
        label_shown = False
        for k in range(int(2*J+1)):
            MJ = k-J
            derivative = derivatives_[states_FJM_index(F, J, MJ, L)]
            # Plot derivative in MHz/mT for better readability
            ax.plot(Bplot, derivative*1e-3, color=col, linestyle='--')
            if not label_shown:
                ax.plot([],[],color=col, label='F={}, J={}'.format(F, J))
                label_shown = True

        ax.set_title(ptit)
        ax.set_xlabel(xlab)
        ax.set_ylabel('Derivative of Eigenenergies (MHz/mT)')
        ax.legend()

        plt.tight_layout()
        if separate:  # Display the figure after each pair if Separate is checked
            plt.show()

    if not separate:  # Display the figure after all pairs if Separate is not checked
        plt.show()

############################################################################################################
#Table of 2 photon transitions from a given (nu_i,Li) to (nu_f,Lf) level of H2+
#For a given magnetic field value B
def table_2photon_transitions(nu_i,Li,nu_f,Lf,B,prob_threshold=1e-3, sensitivity_threshold=None):
    # Get the eigenvalues, eigenvectors and derivatives for the given B
    EWsi_B, EVsi_B,dEi_B_ = get_EWs_EVs_derivatives_B0(nu_i, Li, B)
    EWsf_B, EVsf_B,dEf_B_ = get_EWs_EVs_derivatives_B0(nu_f, Lf, B)
    # Get the number of states
    Ni = len(EWsi_B)
    Nf = len(EWsf_B)
    if redQ(nu_i,Li,nu_f,Lf,0) == None or redQ(nu_i,Li,nu_f,Lf,2) == None:
        print('No 2-photon transitions for the given levels implemented.')
        return

    pi = mat_SQ_mix(0,0,nu_i,Li,EVsi_B,nu_f,Lf,EVsf_B)
    sp_sp = mat_SQ_mix(1,1,nu_i,Li,EVsi_B,nu_f,Lf,EVsf_B)
    sp_sm = mat_SQ_mix(1,-1,nu_i,Li,EVsi_B,nu_f,Lf,EVsf_B)

    # Initialize empty lists to store the transition data
    transitions = []
    for i in range(Ni):
        for j in range(Nf):
            # Calculate the energy difference
            diff_Eij = EWsf_B[j]-EWsi_B[i]
            # Calculate the magnetic field sensitivity
            diff_dE_Bij = dEf_B_[j]-dEi_B_[i]
            # If the transition probability is above the threshold, add the transition data to the list
            if prob_threshold is None or abs(pi[i,j]) > prob_threshold or abs(sp_sp[i,j]) > prob_threshold or abs(sp_sm[i,j]) > prob_threshold:
                if sensitivity_threshold is None or abs(diff_dE_Bij) < sensitivity_threshold:
                    transitions.append([i, j, diff_Eij, diff_dE_Bij, pi[i,j], sp_sp[i,j], sp_sm[i,j]])
    if len(transitions) == 0:
        print('No transitions with the given thresholds.')
        return
    # Create a DataFrame from the list of transitions
    df = pd.DataFrame(transitions, columns=['Initial State', 'Final State', 'Transition frequency [MHz]', \
                            'Magnetic Field Sensitivity [MHz/T]', '$\\pi\\pi$', '$\\sigma_{+}\\sigma_{+}$', '$\\sigma_{+}\\sigma_{-}$'])
    return df

def table_insensitive_2photon_transitions(nu_i,Li,nu_f,Lf,Bmin,Bmax,samples=1000, prob_threshold=1e-3, sensitivity_threshold=1):
    """
    Table of 2-photon transitions for a given (nu_i,Li) to (nu_f,Lf) level of H2+ and the magnetic field value at which the transitions are insensitive to the magnetic field.
    This function calculates the energy difference, magnetic field sensitivity, and transition probabilities for all 2-photon transitions for a given (nu_i,Li) to (nu_f,Lf) level of H2+ and the magnetic field value at which the transitions are insensitive to the magnetic field.

    Parameters
    ----------
    nu_i : int
        The initial nu level.
    Li : int
        The initial L level.
    nu_f : int
        The final nu level.
    Lf : int
        The final L level.
    Bmax : float
        The maximal magnetic field value to consider for the range of magnetic field values.
    samples : int, optional
        The number of samples to use for the magnetic field values. Default is 1000.
    prob_threshold : float, optional
        The probability threshold for showing transitions. Default is 1e-3.
    sensitivity_threshold : float, optional
        The sensitivity threshold for considering a transition insensitive. Default is 1 [MHz/T].

    Returns
    -------
    DataFrame
        The table of 2-photon transitions.
    """
    Bvalues = np.linspace(Bmin, Bmax, samples)
    B_values_zero_sensibility = insensitive_transitions(nu_i,Li,nu_f,Lf,Bvalues,threshold=sensitivity_threshold)

    insensitive_two_photon_transitions = []

    for B, i, f, sensibility_if in B_values_zero_sensibility:
        EWsi_B, EVsi_B = get_EWs_EVs_B0(nu_i, Li, B)
        EWsf_B, EVsf_B = get_EWs_EVs_B0(nu_f, Lf, B)
        pi = mat_SQ_mix(0,0,nu_i,Li,EVsi_B,nu_f,Lf,EVsf_B)
        sp_sp = mat_SQ_mix(1,1,nu_i,Li,EVsi_B,nu_f,Lf,EVsf_B)
        sp_sm = mat_SQ_mix(1,-1,nu_i,Li,EVsi_B,nu_f,Lf,EVsf_B)
        if prob_threshold is None or abs(pi[i,f]) > prob_threshold or abs(sp_sp[i,f]) > prob_threshold or abs(sp_sm[i,f]) > prob_threshold:
            insensitive_two_photon_transitions.append([B, i, f, EWsf_B[f]-EWsi_B[i], sensibility_if, pi[i,f], sp_sp[i,f], sp_sm[i,f]])
    if len(insensitive_two_photon_transitions) == 0:
        print('No insensitive transitions with the given thresholds.')
        return
    display(Markdown(f'The insensitive 2-photon transitions from the rovibrational level $(\\nu_i,\\mathbf{{L}}_i)=({nu_i},{Li})$ to $(\\nu_f,\\mathbf{{L}}_f)=({nu_f},{Lf})$ for magnetic fields up to {Bmax} T are shown in the table below.'))
    df = pd.DataFrame(insensitive_two_photon_transitions, columns=['Magnetic Field [T]', 'Initial State', 'Final State', 'Transition frequency [MHz]', \
                                'Magnetic Field Sensitivity [MHz/T]', '$\\pi\\pi$', '$\\sigma_{+}\\sigma_{+}$', '$\\sigma_{+}\\sigma_{-}$'])
    return df

############################################################################################################
from IPython.display import display, Markdown

# Define a dictionary to map color codes to CSS color names
color_map = {
    'b': 'blue',
    'r': 'red',
    'g': 'green',
    'y': 'yellow',
    'm': 'magenta',
    'c': 'cyan'
}

def color_cells(color):
    # Convert the color code to a CSS color name
    css_color = color_map.get(color, 'black')  # Default to 'black' if the color code is not in the map
    return f'background-color: {css_color}; color: {css_color}'

# Function to convert a number to LaTeX fraction format
def to_latex_fraction(num, expl_plus=False):
    frac = Fraction(num).limit_denominator()
    if frac < 0:
        return r'$-\frac{{{}}}{{{}}}$'.format(abs(frac.numerator), frac.denominator)
    else:
        if expl_plus:
            return r'$+\frac{{{}}}{{{}}}$'.format(frac.numerator, frac.denominator) 
        return r'$\frac{{{}}}{{{}}}$'.format(frac.numerator, frac.denominator) if frac.denominator != 1 else str(frac.numerator)

#table explaining content of plot
def display_table(nu,L):
    states_ = states_FJM(L)
    N_ = len(states_)
    numberofstate = range(N_)
    colors = [lev_col(L,*states_[j][:2]) for j in numberofstate]

    #Explaining content of table
    display(Markdown(f'In each plot a column represents the coefficient squared of the eigenstate\
                      corresponding to the basis state $|F, J, M_J \\rangle$.\
                     The color of the column corresponds to the $F$,$J$ state of the bar.'))

    # Create a DataFrame for the table
    pd.set_option('display.html.use_mathjax', True)
    data = {
        'Column': [i for i in numberofstate],
        '$F$': [to_latex_fraction(states_[i][0]) for i in numberofstate],
        '$J$': [to_latex_fraction(states_[i][1]) for i in numberofstate],
        '$M_J$': [to_latex_fraction(states_[i][2]) for i in numberofstate],
        'Color': colors
    }
    df = pd.DataFrame(data)

    # Transpose the DataFrame
    df = df.T

    # Create a Styler object from the DataFrame
    df_styled = df.style.apply(lambda x: ['background-color: {}; color: {}'.format(color_map.get(val, 'black'), \
                                        color_map.get(val, 'black')) if x.name == 'Color' else '' for val in x], axis=1)
    # Add borders to each cell
    df_styled = df_styled.set_table_styles([
       {
         'selector': 'td',
         'props': [('border', '1px solid black'), ('text-align', 'center'), ('width', '100px')]
       }
    ])

    display(df_styled)  # Display the styled DataFrame

# Function to plot eigenvectors for a given magnetic field value
def plot_eigenvectors(L,numberofstate_,EWs_,EVs_, B, scale):
    states_ = states_FJM(L)
    range_of_N = range(len(numberofstate_))

    colors = [lev_col(*states_[j][:2]) for j in range_of_N]

    #rescale B field when printing
    if B<=1e-3:
        Bprint = B*1e6
        unit = 'µT'
    elif B<=1:
        Bprint = B*1e3
        unit = 'mT'
    else:
        Bprint = B
        unit = 'T'

    # Create a list to store the figures and axes
    figs_axes = []

    for i in numberofstate_:
        # Create a new figure with a specific size
        fig, ax = plt.subplots(figsize=(10, 6))

        c_squared = EVs_[:, i]**2

        ax.set_title(f'State {i} of energy E = {EWs_[i]} MHz')
        ax.bar(range_of_N, c_squared, color=colors)
        ax.set_xlabel('Component')
        ax.set_ylabel(f'Amplitude Squared in {scale} scale')
        ax.set_xticks(range_of_N)
        ax.set_ylim([0, 1])  # Set y-axis limits

        # Create a custom legend
        color_legend = {lev_col(F, J): f'({F}, {J})' for F, J, _ in states_}
        handles = [plt.Rectangle((0,0),1,1, color=color) for color in color_legend.keys()]
        labels = list(color_legend.values())
        legend = plt.legend(handles, labels)
        # Add a title to the legend
        legend.set_title("Color represents (F, J)")

        if scale == 'log':
            plt.yscale('log')
            plt.ylim([10**-10, plt.gca().get_ylim()[1]])  # Set y-axis limits for logarithmic scale
        #assure that the scale is either linear or logarithmic
        elif scale != 'linear':
            raise ValueError("scale must be either 'linear' or 'log'")
        
        # Add the figure and axes to the list
        figs_axes.append([fig, ax])
    
    # Return the list of figures and axes
    return figs_axes


def table_eigenstates(nu,L,B):
    states_ = states_FJM(L)
    N_ = len(states_)
    numberofstate = range(N_)

    EWs, EVs = get_EWs_EVs_B0(nu, L, B)

    colors = [lev_col(L,*states_[j][:2]) for j in numberofstate]

    # Create a DataFrame for the table
    pd.set_option('display.html.use_mathjax', True)
    data = {
        'Column': [i for i in numberofstate],
        '$F$': [to_latex_fraction(states_[i][0]) for i in numberofstate],
        '$J$': [to_latex_fraction(states_[i][1]) for i in numberofstate],
        '$M_J$': [to_latex_fraction(states_[i][2]) for i in numberofstate],
        'Color': colors,
        'Energy [MHz]': [EWs[i] for i in numberofstate],
        'Coefficient Squared': [EVs[i,i]**2 for i in numberofstate]
    }
    df = pd.DataFrame(data)

    # Transpose the DataFrame
    df = df.T

    # Create a Styler object from the DataFrame
    df_styled = df.style.apply(lambda x: ['background-color: {}; color: {}'.format(color_map.get(val, 'black'), \
                                        color_map.get(val, 'black')) if x.name == 'Color' else '' for val in x], axis=1)
    # Add borders to each cell
    df_styled = df_styled.set_table_styles([
       {
         'selector': 'td',
         'props': [('border', '1px solid black'), ('text-align', 'center'), ('width', '100px')]
       }
    ])

    display(df_styled)  # Display the styled DataFrame

############################################################################################################
############################################################################################################
#Functions to obtain Latex tables

def hf_indices_latex(L):
    states = states_FJM(L)
    table = []
    for i, state in enumerate(states):
        F, J, MJ = state
        F,J,MJ = to_latex_fraction(F), to_latex_fraction(J), to_latex_fraction(MJ,True)
        row = [i, F, J, MJ]
        table.append(row)
    headers = ["State index", f"\\multicolumn{{3}}{{c}}{{($F,J,M_J$)}}"]
    latex_table = "\\begin{tabular}{c|ccc}\n"
    latex_table += f"\\multicolumn{{4}}{{c}}{{$L = {L}$}} \\\\ \hline\n"  # First row where L stands
    latex_table += " & ".join(headers) + "\\\\ \hline\n"
    for row in table:
        latex_table += " & ".join(str(x) for x in row) + "\\\\ \n"
    latex_table += "\\end{tabular}"
    print(latex_table)

def hf_indices_energies_B_latex(nu,L,B):
    if B<=1e-6:
        Bheader = B*1e9
        Bunits = "nT"
    elif B<=1e-3:
        Bheader = B*1e6
        Bunits = "$\mu$T"
    elif B<=1:
        Bheader = B*1e3
        Bunits = "mT"
    else:
        Bheader = B
        Bunits = "T"

    states = states_FJM(L)
    EWs_B, EVs_B = get_EWs_EVs_B0(nu, L, B)
    table = []
    for i, state in enumerate(states):
        F, J, MJ = state
        F,J,MJ = to_latex_fraction(F), to_latex_fraction(J), to_latex_fraction(MJ)
        c_i_squared = round(np.abs(EVs_B[i,i])**2, 5)
        E_i = round(EWs_B[i], 5)
        row = [i, f"({F},{J},{MJ})", E_i, c_i_squared]
        table.append(row)
    headers = ["State index", "($F,J,M_J$)", "Energy [MHz]", "$|c_i|^2$"]
    latex_table = "\\begin{tabular}{c|c|c|c}\n"
    latex_table += f"\\multicolumn{{2}}{{c|}}{{$(\\nu,L) = ({nu,L})$}} & \\multicolumn{{2}}{{c}}{{B={Bheader} [{Bunits}]}} \\\\ \hline\n"  # First row where L stands
    latex_table += " & ".join(headers) + "\\\\ \hline\n"
    for row in table:
        latex_table += " & ".join(str(x) for x in row) + "\\\\ \n"
    latex_table += "\\end{tabular}"
    print(latex_table)

def insensitive_transitions_latex_same(nu,L,Bmin,Bmax,points=1000, sensitivity=1):
    Bvalues = np.linspace(Bmin,Bmax,points)
    transitions = insensitive_transitions_same(nu,L,Bvalues,sensitivity)
    
    # transitions is a list of tuples (B, i, f, sensitivity)
    table = []
    
    B_header = "B [T]"
    if Bmax<=1e-3:
        B_header = "B [$\mu$T]"
    elif Bmax<=1:
        B_header = "B [mT]"
    
    for transition in transitions:
        B, i, f, s = transition
        # tuple (F, J, MJ)
        states = states_FJM(L)
        F_i, J_i, MJ_i = states[i]
        F_f, J_f, MJ_f = states[f]
        # Convert F, J, MJ to LaTeX fractions
        F_i,J_i,MJ_i = to_latex_fraction(F_i), to_latex_fraction(J_i), to_latex_fraction(MJ_i)
        F_f,J_f,MJ_f = to_latex_fraction(F_f), to_latex_fraction(J_f), to_latex_fraction(MJ_f)
        
        EB,_ = get_EWs_EVs_B0(nu, L, B)
        
        DE = EB[f] - EB[i]
        DE = "{:.5f}".format(DE)  # Format DE to have 5 digits after the decimal point
        
        B_table = round(B,12)
        B_header = "B [T]"
        if Bmax<=1e-3:
            B_table = round(B*1e6,4)
        elif Bmax<=1:
            B_table = round(B*1e3,8)
        
        row = [i, f"({F_i},{J_i},{MJ_i})", f, f"({F_f},{J_f},{MJ_f})", DE, B_table]
        table.append(row)

    headers = ["$i$", "($F,J,M_J$)", "$f$", "($F,J,M_J$)", "Frequency [MHz]", B_header]
    latex_table = "\\begin{tabular}{cc|cc|c|c}\n" + " & ".join(headers) + "\\\\ \hline\n"
    for row in table:
        latex_table += " & ".join(str(x) for x in row) + "\\\\ \n"
    latex_table += "\\end{tabular}"
    print(latex_table)

def insensitive_transitions_latex(nu_i,Li,nu_f,Lf,Bmin,Bmax,points=1000, sensitivity=1,with_sensitivity=False,inlatextable=False):
    Bvalues = np.linspace(Bmin,Bmax,points)
    transitions = insensitive_transitions(nu_i,Li,nu_f,Lf,Bvalues,sensitivity)
    DE_levels = Energy_diff(nu_i,Li,nu_f,Lf)
    # transitions is a list of tuples (B, i, f, sensitivity)
    table = []

    B_header = "B [T]"
    if Bmax<=1e-3:
        B_header = "B [$\mu$T]"
    elif Bmax<=1:
        B_header = "B [mT]"

    for transition in transitions:
        B, i, f, s = transition
        #tuple (F, J, MJ)
        F_i, J_i, MJ_i = states_FJM(Li)[i]
        F_f, J_f, MJ_f = states_FJM(Lf)[f]
        # Convert F, J, MJ to LaTeX fractions
        F_i,J_i,MJ_i = to_latex_fraction(F_i), to_latex_fraction(J_i), to_latex_fraction(MJ_i,True)
        F_f,J_f,MJ_f = to_latex_fraction(F_f), to_latex_fraction(J_f), to_latex_fraction(MJ_f,True)

        EB_init,_ = get_EWs_EVs_B0(nu_i, Li, B)
        EB_final,_ = get_EWs_EVs_B0(nu_f, Lf, B)
        DE = EB_final[f] - EB_init[i]
        DE = "{:.5f}".format(DE)  # Format DE to have 5 digits after the decimal point
        if Bmax<=1e-3:
            B_table = round(B*1e6,4)
        elif Bmax<=1:
            B_table = round(B*1e3,8)
        else:
            B_table = round(B,12)
        if with_sensitivity:
            row = [i, f"({F_i},{J_i},{MJ_i})", f, f"({F_f},{J_f},{MJ_f})", DE, B_table, round(s,2)]
        else:
            row = [i, f"({F_i},{J_i},{MJ_i})", f, f"({F_f},{J_f},{MJ_f})", DE, B_table]
        table.append(row)
    
    headers1 = [f"$(\\nu_i, L_i)=({nu_i},{Li})$", "", f"$(\\nu_f, L_f)=({nu_f},{Lf})$", "",f"$\Delta E$={DE_levels}"]
    if with_sensitivity:
        headers2 = ["$i$", "($F,J,M_J$)", "$f$", "($F,J,M_J$)", "Frequency [MHz]", B_header, "Sensitivity [MHz/T]"]
        shape = "cc|cc|c|c|c"
    else:
        headers2 = ["$i$", "($F,J,M_J$)", "$f$", "($F,J,M_J$)", "Frequency shift [MHz]", B_header]
        shape = "cc|cc|c|c"
    
    latex_table = ""

    if inlatextable:
        latex_table += "\\begin{table}[ht]\n"
        latex_table += "\\centering\n"
        latex_table += "\\resizebox{\\textwidth}{!}{%\n"

    latex_table += "\\begin{tabular}{" + shape + "}\n" + " & ".join(headers1) + "\\\\ \n" + " & ".join(headers2) + "\\\\ \hline\n"
    for row in table:
        latex_table += " & ".join(str(x) for x in row) + "\\\\ \n"
    latex_table += "\\end{tabular}"

    if inlatextable:
        latex_table += "}\n"
        latex_table += "\\caption{Caption}\n"
        latex_table += "\\label{tab:my_label}\n"
        latex_table += "\\end{table}"

    print(latex_table)

def insensitive_transitions_ranges_latex(nu_i,Li,nu_f,Lf,Bmin,Bmax,points=1000, sensitivity=1,with_sensitivity=False,inlatextable=False):
    states_i = states_FJM(Li)
    states_f = states_FJM(Lf)

    Bvalues = np.linspace(Bmin,Bmax,points)
    
    EWs_i, EVs_i, DEB_i = get_EWs_EVs_derivatives(nu_i,Li,Bvalues)
    if (nu_i,Li)==(nu_f,Lf):
        EWs_f, EVs_f, DEB_f = EWs_i, EVs_i, DEB_i
    else:
        EWs_f, EVs_f, DEB_f = get_EWs_EVs_derivatives(nu_f,Lf,Bvalues)
    
    # transitions is a list of tuples (i,f,Brange,min_sensibility,B_min_sensibility)
    transitions = insensitive_transitions_ranges(nu_i,Li,DEB_i,nu_f,Lf,DEB_f,Bvalues,sensitivity)
    diff_E_levels = Energy_diff(nu_i,Li,nu_f,Lf,rel=False)
    table = []

    Bmin_header = "B_{{min}} [T]"
    Bmax_header = "B_{{max}} [T]"
    Bsen_header = "B_{{sens}} [T]"
    if Bmax<=1e-3:
        Bmin_header = "B_{{min}} [$\mu$T]"
        Bmax_header = "B_{{max}} [$\mu$T]"
        Bsen_header = "B_{{sens}} [$\mu$T]"
    elif Bmax<=1:
        Bmin_header = "B_{{min}} [mT]"
        Bmax_header = "B_{{max}} [mT]"
        Bsen_header = "B_{{sens}} [mT]"
    
    for transition in transitions:
        i, f, Brange, min_sensibility, Bsen, index_Bsen = transition
        Bmin_tr, Bmax_tr = Brange
        #tuple (F, J, MJ)
        F_i, J_i, MJ_i = states_i[i]
        F_f, J_f, MJ_f = states_f[f]
        # Convert F, J, MJ to LaTeX fractions
        F_i,J_i,MJ_i = to_latex_fraction(F_i), to_latex_fraction(J_i), to_latex_fraction(MJ_i,True)
        F_f,J_f,MJ_f = to_latex_fraction(F_f), to_latex_fraction(J_f), to_latex_fraction(MJ_f,True)

        DE = round(EWs_f[f,index_Bsen] - EWs_i[i,index_Bsen],4)

        if Bmax<=1e-3:
            Bsen_table = round(Bsen*1e6,4)
            Bmin_table = round(Bmin_tr*1e6,4)
            Bmax_table = round(Bmax_tr*1e6,4)
        elif Bmax<=1:
            Bsen_table = round(Bsen*1e3,8)
            Bmin_table = round(Bmin_tr*1e3,8)
            Bmax_table = round(Bmax_tr*1e3,8)
        else:
            Bsen_table = round(Bsen,9)
            Bmin_table = round(Bmin_tr,9)
            Bmax_table = round(Bmax_tr,9)
        
        if with_sensitivity:
            row = [i, F_i,J_i,MJ_i, f, F_f,J_f,MJ_f, DE, Bmin_table, Bmax_table, Bsen_table, round(min_sensibility,4)]
        else:
            row = [i, F_i,J_i,MJ_i, f, F_f,J_f,MJ_f, DE, Bmin_table, Bmax_table]
        table.append(row)
    
    headers1 = [f"$(\\nu_i, L_i)=({nu_i},{Li})$", "", "", "", f"$(\\nu_f, L_f)=({nu_f},{Lf})$", "", "", "",f"$\Delta E$={diff_E_levels}"]
    if with_sensitivity:
        headers2 = ["$i$", "\\multicolumn{3}{c|}{($F,J,M_J$)}", "$f$", "\\multicolumn{3}{c|}{($F,J,M_J$)}", "$\\Delta f_{{hf}}$ [MHz]", Bmin_header, Bmax_header, Bsen_header, "Min sens. [MHz/T]"]
        shape = "c|ccc|c|ccc|c|cc|cc"
    else:
        headers2 = ["$i$", "\\multicolumn{3}{($F,J,M_J$)}", "$f$", "\\multicolumn{3}{($F,J,M_J$)}", "$\\Delta f_{{hf}}$ [MHz]", Bmin_header, Bmax_header]
        shape = "c|ccc|c|ccc|c|cc"
    
    latex_table = ""

    if inlatextable:
        latex_table += "\\begin{table}[ht]\n"
        latex_table += "\\centering\n"
        latex_table += "\\resizebox{\\textwidth}{!}{%\n"

    latex_table += "\\begin{tabular}{" + shape + "}\n" + " & ".join(headers1) + "\\\\ \n" + " & ".join(headers2) + "\\\\ \hline\n"
    for row in table:
        latex_table += " & ".join(str(x) for x in row) + "\\\\ \n"
    latex_table += "\\end{tabular}"

    if inlatextable:
        latex_table += "}\n"
        latex_table += "\\caption{Caption}\n"
        latex_table += "\\label{tab:my_label}\n"
        latex_table += "\\end{table}"

    print(latex_table)

def insensitive_mag_dipole_transitions_latex_all_mu_q(nu,L,Bmin,Bmax,samples=1000,prob_threshold=1e-5, sensitivity_threshold=10,inlatextable=False):
    """
    Table of magnetic dipole transitions for a given (nu,L) level of H2+ and the magnetic field value at which the transitions are insensitive to the magnetic field.
    This function calculates the energy difference, magnetic field sensitivity, and transition probabilities for all magnetic dipole transitions for a given (nu,L) level of H2+ and the magnetic field value at which the transitions are insensitive to the magnetic field.

    Parameters
    ----------
    nu : int
        The nu level.
    L : int
        The L level.
    Bmax : float
        The maximal magnetic field value to consider for the range of magnetic field values.
    samples : int, optional
        The number of samples to use for the magnetic field values. Default is 1000.
    prob_threshold : float, optional
        The probability threshold for showing transitions. Default is 1e-3.
    sensitivity_threshold : float, optional
        The sensitivity threshold for considering a transition insensitive. Default is 10 [MHz/T].

    Returns
    -------
    DataFrame
        The table of magnetic dipole transitions.
    """
    
    Bvalues = np.linspace(Bmin, Bmax, samples)
    EWs,EVs,DEB = get_EWs_EVs_derivatives(nu,L,Bvalues)

    Bsen_header = "B [T]"
    if Bmax<=1e-3:
        Bsen_header = "B [$\mu$T]"
    elif Bmax<=1:
        Bsen_header = "B [mT]"

    transitions = insensitive_transitions_ranges(nu,L,DEB,nu,L,DEB,Bvalues,sensitivity_threshold)

    insensitive_M1_transitions = []
    
    for transition in transitions:
        i, f, Brange, min_sensibility, Bsen, index_Bsen = transition
        EVs_B = EVs[index_Bsen]
        mu_p = mu_div_mub_mixed(1, nu, L,EVs=EVs_B)
        mu_0 = mu_div_mub_mixed(0, nu, L,EVs=EVs_B)
        mu_m = mu_div_mub_mixed(-1, nu, L,EVs=EVs_B)

        DE_if = round(EWs[f,index_Bsen] - EWs[i,index_Bsen],4)

        if prob_threshold is None or abs(mu_p[f,i]) > prob_threshold or abs(mu_0[f,i]) > prob_threshold or abs(mu_m[f,i]) > prob_threshold:
            insensitive_M1_transitions.append([i, f, DE_if, Bsen, round(mu_p[f,i],5), round(mu_0[f,i],5), round(mu_m[f,i],5)])
        
    headers1 = [f"$(\\nu, L)=({nu},{L})$", "", "", "", "", ""]
    headers2 = ["$i$", "$f$", "Frequency shift [MHz]", Bsen_header, "$\mu_+$", "$\mu_0$", "$\mu_-$"]
    shape = "c|c|c|c|ccc"

    latex_table = ""

    if inlatextable:
        latex_table += "\\begin{table}[ht]\n"
        latex_table += "\\centering\n"
        latex_table += "\\resizebox{\\textwidth}{!}{%\n"

    latex_table += "\\begin{tabular}{" + shape + "}\n" + " & ".join(headers1) + "\\\\ \n" + " & ".join(headers2) + "\\\\ \hline\n"
    for row in insensitive_M1_transitions:
        latex_table += " & ".join(str(x) for x in row) + "\\\\ \n"
    latex_table += "\\end{tabular}"

    if inlatextable:
        latex_table += "}\n"
        latex_table += "\\caption{Caption}\n"
        latex_table += "\\label{tab:my_label}\n"
        latex_table += "\\end{table}"

    print(latex_table)

def insensitive_quadrupole_transitions_onetheta_latex(nu_i,Li,nu_f,Lf,Bmin,Bmax,samples=1001,prob_threshold=1e-5, sensitivity_threshold=10,inlatextable=False):
    
    Bvalues = np.linspace(Bmin, Bmax, samples)
    EWs_i, EVs_i, DEB_i = get_EWs_EVs_derivatives(nu_i,Li,Bvalues)
    states_i = states_FJM(Li)
    EWs_f, EVs_f, DEB_f = get_EWs_EVs_derivatives(nu_f,Lf,Bvalues)
    states_f = states_FJM(Lf)

    DE_levels = Energy_diff(nu_i,Li,nu_f,Lf)

    Bsen_header = "B [T]"
    Bs_table = Bvalues
    if Bmax<=1e-3:
        Bsen_header = "B [$\mu$T]"
        Bs_table = Bvalues*1e6
    elif Bmax<=1:
        Bsen_header = "B [mT]"
        Bs_table = Bvalues*1e3

    transitions = insensitive_transitions_ranges(nu_i,Li,DEB_i,nu_f,Lf,DEB_f,Bvalues,sensitivity_threshold)

    insensitive_E2_transitions = []

    Theta_m2 = mat_Theta(-2,nu_f,Lf,nu_i,Li)
    Theta_m1 = mat_Theta(-1,nu_f,Lf,nu_i,Li)
    Theta_0 = mat_Theta(0,nu_f,Lf,nu_i,Li)
    Theta_p1 = mat_Theta(1,nu_f,Lf,nu_i,Li)
    Theta_p2 = mat_Theta(2,nu_f,Lf,nu_i,Li)
    
    def Theta(q):
        if q == -2:
            return Theta_m2
        elif q == -1:
            return Theta_m1
        elif q == 0:
            return Theta_0
        elif q == 1:
            return Theta_p1
        elif q == 2:
            return Theta_p2

    for transition in transitions:
        i, f, Brange, min_sensibility, B, index_B = transition
        EVs_i_B = EVs_i[:,index_B]
        EVs_f_B = EVs_f[:,index_B]

        B_table = round(Bs_table[index_B],6)

        F_i,J_i,MJ_i = states_i[i]
        F_f,J_f,MJ_f = states_f[f]
        latex_MJ_i, latex_MJ_f = to_latex_fraction(MJ_i,True), to_latex_fraction(MJ_f,True)
        q = int(MJ_i - MJ_f)
        if abs(q)<=2:
            Tqif = np.einsum('ik,kl,jl->ij',EVs_f_B,Theta(q),EVs_i_B)[f,i]
        else:
            Tqif = 0

        DE_if = round(EWs_f[f,index_B] - EWs_i[i,index_B],4)

        if prob_threshold is None or abs(Tqif) > prob_threshold:
            insensitive_E2_transitions.append([i, latex_MJ_i, f, latex_MJ_f, DE_if, B_table, q, str(round(Tqif*1e3,5)) + "(-3)"])
        
    headers1 = [f"$(\\nu_i, L_i)=({nu_i},{Li})$", "", f"$(\\nu_f, L_f)=({nu_f},{Lf})$", "", f"$\\Delta E = {DE_levels}$ THz", "", "", ""]
    headers2 = ["i", "$M_J$", "f", "$M_J'", "Frequency shift [MHz]", Bsen_header, "$q$", "$T_q$"]
    shape = "cc|cc|c|c|cc"

    latex_table = ""

    if inlatextable:
        latex_table += "\\begin{table}[ht]\n"
        latex_table += "\\centering\n"
        latex_table += "\\resizebox{\\textwidth}{!}{%\n"

    latex_table += "\\begin{tabular}{" + shape + "}\n" + " & ".join(headers1) + "\\\\ \n" + " & ".join(headers2) + "\\\\ \hline\n"
    for row in insensitive_E2_transitions:
        latex_table += " & ".join(str(x) for x in row) + "\\\\ \n"
    latex_table += "\\end{tabular}"

    if inlatextable:
        latex_table += "}\n"
        latex_table += "\\caption{Caption}\n"
        latex_table += "\\label{tab:my_label}\n"
        latex_table += "\\end{table}"

    print(latex_table)

def insensitive_twophoton_transitions_latex(nu_i,Li,nu_f,Lf,Bmin,Bmax,samples=1001,prob_threshold=1e-5, sensitivity_threshold=10,pulsetime=0.1,theta=0):
    Bvalues = np.linspace(Bmin, Bmax, samples)
    EWs_i, EVs_i, DEB_i = get_EWs_EVs_derivatives(nu_i,Li,Bvalues)
    states_i = states_FJM(Li)
    EWs_f, EVs_f, DEB_f = get_EWs_EVs_derivatives(nu_f,Lf,Bvalues)
    states_f = states_FJM(Lf)

    DE_levels = Energy_diff(nu_i,Li,nu_f,Lf)


    Bsen_header = "B [T]"
    Bs_table = Bvalues
    if Bmax<=1e-3:
        Bsen_header = "B [$\mu$T]"
        Bs_table = Bvalues*1e6
    elif Bmax<=1:
        Bsen_header = "B [mT]"
        Bs_table = Bvalues*1e3

    atheta_i = alpha_theta(nu_i,Li,0,theta,False)
    atheta_f = alpha_theta(nu_f,Lf,0,theta,False)

    transitions = insensitive_transitions_ranges(nu_i,Li,DEB_i,nu_f,Lf,DEB_f,Bvalues,sensitivity_threshold)

    insensitive_2E1_transitions = []

    SQ_m = mat_SQ(-1,-1,nu_f,Lf,nu_i,Li)
    SQ_0 = mat_SQ(0,0,nu_f,Lf,nu_i,Li)
    SQ_p = mat_SQ(1,1,nu_f,Lf,nu_i,Li)

    def SQ(q):
        if q == -2:
            return SQ_m
        elif q == 0:
            return SQ_0
        elif q == 2:
            return SQ_p
        else:
            return None

    for transition in transitions:
        i, f, Brange, min_sensibility, B, index_B = transition
        EVs_i_B = EVs_i[:,index_B]
        EVs_f_B = EVs_f[:,index_B]

        B_table = round(Bs_table[index_B],6)

        F_i,J_i,MJ_i = states_i[i]
        F_f,J_f,MJ_f = states_f[f]
        latex_MJ_i, latex_MJ_f = to_latex_fraction(MJ_i,True), to_latex_fraction(MJ_f,True)

        delta_alpha = atheta_f[f,f] - atheta_i[i,i]

        q = int(MJ_i - MJ_f)
        if SQ(q) is not None:
            S_qif = np.einsum('ik,kl,jl->ij',EVs_f_B,SQ(q),EVs_i_B)[f,i]
            AC_shift = -1/(4*pulsetime) * delta_alpha/abs(S_qif)
        else:
            S_qif = 0

        DE_if = round(EWs_f[f,index_B] - EWs_i[i,index_B],4)

        if prob_threshold is None or abs(S_qif) > prob_threshold:
            insensitive_2E1_transitions.append([i, latex_MJ_i, f, latex_MJ_f, B_table, round(delta_alpha,5), q, round(S_qif,5), DE_if, round(AC_shift,3)])
        
    headers1 = ["\multicolumn{2}{|c|}{$(\\nu, L)="+f"({nu_i},{Li})"+"$}", "\multicolumn{2}{c|}{$(\\nu, L)="+f"({nu_f},{Lf})"+"$}", "", "\multicolumn{3}{c|}{}", "\multicolumn{2}{c|}{" + f"$f_0 = {DE_levels}$ MHz"+ "}"]
    headers2 = ["i", "$M_J$", "f", "$M_J'$", Bsen_header, "$\Delta \\alpha_{if}$", "$q$", "$^SQ_{qq}$", "$\Delta f_{hf}$ [MHz]", "$\Delta f_{LS}$ [Hz]"]
    shape = "cc|cc|c|ccc|cc"

    latex_table = ""

    

    latex_table += "\\begin{longtable}{" + shape + "}\n" + " & ".join(headers1) + "\\\\ \n" + " & ".join(headers2) + "\\\\ \hline\n"
    for row in insensitive_2E1_transitions:
        latex_table += " & ".join(str(x) for x in row) + "\\\\ \n"
    latex_table += "\\end{longtable}"

    print(latex_table)

############################################################################################################

def insensitive_magnetic_dipole_transitions_latex(nu,L,Bmin,Bmax,samples=1001,prob_threshold=1e-6, sensitivity_threshold=10):
    Bvalues = np.linspace(Bmin, Bmax, samples)
    EWs, EVs, DEB = get_EWs_EVs_derivatives(nu,L,Bvalues)
    states = states_FJM(L)

    transitions = insensitive_transitions_ranges(nu,L,DEB,nu,L,DEB,Bvalues,sensitivity_threshold)

    insensitive_M1_transitions = []

    Bsen_header = "B [T]"
    Bs_table = Bvalues
    if Bmax<=1e-3:
        Bsen_header = "B [$\mu$T]"
        Bs_table = Bvalues*1e6
    elif Bmax<=1:
        Bsen_header = "B [mT]"
        Bs_table = Bvalues*1e3

    mu_m1 = mu_div_mub(-1,nu,L)
    mu_0 = mu_div_mub(0,nu,L)
    mu_p1 = mu_div_mub(1,nu,L)
    def mu(q):
        if q == -1:
            return mu_m1
        elif q == 0:
            return mu_0
        elif q == 1:
            return mu_p1
    
    for transition in transitions:
        i, f, Brange, min_sensibility, B, index_B = transition
        EVs_i_B = EVs[:,index_B]
        EVs_f_B = EVs[:,index_B]

        F_i,J_i,MJ_i = states[i]
        F_f,J_f,MJ_f = states[f]
        latex_MJ_i, latex_MJ_f = to_latex_fraction(MJ_i,True), to_latex_fraction(MJ_f,True)
        latex_F_i, latex_F_f = to_latex_fraction(F_i,True), to_latex_fraction(F_f,True)
        latex_J_i, latex_J_f = to_latex_fraction(J_i,True), to_latex_fraction(J_f,True)


        q = int(MJ_f - MJ_i)
        if abs(q)<=1:
            Mqif = np.einsum('ik,kl,jl->ij',EVs_f_B,mu(q),EVs_i_B)[f,i]
        else:
            Mqif = 0
        
        DE_if = round(EWs[f,index_B] - EWs[i,index_B],4)

        if abs(Mqif) > prob_threshold:
            insensitive_M1_transitions.append([i, latex_F_i,latex_J_i,latex_MJ_i, f, latex_F_f,latex_J_f, latex_MJ_f, round(Bs_table[index_B],6), q, round(Mqif,5), DE_if])
    
    headers1 = ["\multicolumn{8}{|c|}{$(\\nu, L)="+f"({nu},{L})"+"$}", "", "\multicolumn{2}{c|}{}", ""]
    headers2 = ["$i$", "$F$", "$J$","$M_J$", "$f$", "$F'$", "$J'$", "$M_J'$", "B [T]", "$q$", "$\\bra{f}\mu_q/\mu_B\ket{i}$", "$f_{hf}$ [MHz]"]

    shape = "|cccc|cccc|c|c|c|c|"

    latex_table = "\\begin{longtable}{" + shape + "}\n" + " \hline\n" + " & ".join(headers1) + "\\\\ \n" + " & ".join(headers2) + "\\\\ \hline\n"
    
    for row in insensitive_M1_transitions:
        latex_table += " & ".join(str(x) for x in row) + "\\\\ \n"
    latex_table += "\hline\n"+"\\end{longtable}"

    print(latex_table)

def insensitive_two_photon_transitions_with_shift_latex(nu_i,Li,nu_f,Lf,Bmin,Bmax,samples=1001,theta=0,pulsetime=0.1,sensitivity_threshold=10):
    Bvalues = np.linspace(Bmin, Bmax, samples)
    EWs_i, EVs_i, DEB_i = get_EWs_EVs_derivatives(nu_i,Li,Bvalues)
    states_i = states_FJM(Li)
    EWs_f, EVs_f, DEB_f = get_EWs_EVs_derivatives(nu_f,Lf,Bvalues)
    states_f = states_FJM(Lf)

    DE_levels = Energy_diff(nu_i,Li,nu_f,Lf)

    Bsen_header = "B [T]"
    Bs_table = Bvalues
    if Bmax<=1e-3:
        Bsen_header = "B [$\mu$T]"
        Bs_table = Bvalues*1e6
    elif Bmax<=1:
        Bsen_header = "B [mT]"
        Bs_table = Bvalues*1e3

    transitions = insensitive_transitions_ranges(nu_i,Li,DEB_i,nu_f,Lf,DEB_f,Bvalues,sensitivity_threshold)

    insensitive_2E1_transitions = []

    SQ_m = mat_SQ(-1,-1,nu_i,Li,nu_f,Lf)
    SQ_0 = mat_SQ(0,0,nu_i,Li,nu_f,Lf)
    SQ_p = mat_SQ(1,1,nu_i,Li,nu_f,Lf)

    atheta_i = alpha_theta(nu_i,Li,0,theta,False)
    atheta_f = alpha_theta(nu_f,Lf,0,theta,False)

    def SQ(q):
        if q == -2:
            return SQ_m
        elif q == 0:
            return SQ_0
        elif q == 2:
            return SQ_p
        else:
            return None

    for transition in transitions:
        i, f, Brange, min_sensibility, B, index_B = transition
        EVs_i_B = EVs_i[:,index_B]
        EVs_f_B = EVs_f[:,index_B]

        B_table = round(Bs_table[index_B],6)

        F_i,J_i,MJ_i = states_i[i]
        F_f,J_f,MJ_f = states_f[f]
        latex_MJ_i, latex_MJ_f = to_latex_fraction(MJ_i,True), to_latex_fraction(MJ_f,True)
        q = int(MJ_f - MJ_i)
        if SQ(q) is not None:
            S_qif = np.einsum('ik,kl,jl->ij',EVs_i_B,SQ(-q),EVs_f_B)[i,f]#since <i|SQ|f> = <f|SQ|i>*
            atheta_i_B = np.einsum('ik,kl,jl->ij',EVs_i_B,atheta_i,EVs_i_B)[i,i]
            atheta_f_B = np.einsum('ik,kl,jl->ij',EVs_f_B,atheta_f,EVs_f_B)[f,f]
            delta_alpha = atheta_f_B - atheta_i_B
        else:
            S_qif = 0

        DE_if = round(EWs_f[f,index_B] - EWs_i[i,index_B],4)

        if abs(S_qif) > 1e-5:
            shift = -1/(8*pulsetime) * delta_alpha/abs(S_qif)
            row = [i,latex_MJ_i, f, latex_MJ_f, B_table, round(delta_alpha,4), q, round(abs(S_qif),5), DE_if/2, round(shift,3)]
        else:
            shift = "-"
            row = [i, latex_MJ_i, f, latex_MJ_f, B_table, round(delta_alpha,4), q, round(abs(S_qif),5), DE_if/2, shift]
        insensitive_2E1_transitions.append(row)

    headers1 = ["\multicolumn{2}{c|}{$(\\nu, L)="+f"({nu_i},{Li})"+"$}", "\multicolumn{2}{c|}{$(\\nu, L)="+f"({nu_f},{Lf})"+"$}", "", "", "\multicolumn{2}{c|}{}","\multicolumn{2}{c}{" + f"$f_0 = {DE_levels/2}$ MHz"+ "}"]
    headers2 = ["$i$", "$M_J$", "$f$", "$M_J'$", "$B$ [$\mu$T]", "$\Delta \\alpha$", "q", "$|^SQ_{qq}|$", "\Delta f_{hf} [MHz]", "$\Delta f_{AC}$ [Hz]"]
    shape = "cc|cc|c|c|cc|cc"

    latex_table = "\\begin{tabular}{" + shape + "}\n" + " & ".join(headers1) + "\\\\ \n" + " & ".join(headers2) + "\\\\ \hline\n"
    for row in insensitive_2E1_transitions:
        latex_table += " & ".join(str(x) for x in row) + "\\\\ \n"
    latex_table += "\\end{tabular}"
    print(latex_table)

def insensitive_quadrupole_transitions_with_shift_latex(nu_i,Li,nu_f,Lf,Bmin,Bmax,samples=1001,prob_threshold=1e-6, sensitivity_threshold=10, pulsetime=0.1, show_Tifen=True):
    Bvalues = np.linspace(Bmin, Bmax, samples)
    EWs_i, EVs_i, DEB_i = get_EWs_EVs_derivatives(nu_i,Li,Bvalues)
    states_i = states_FJM(Li)
    EWs_f, EVs_f, DEB_f = get_EWs_EVs_derivatives(nu_f,Lf,Bvalues)
    states_f = states_FJM(Lf)

    DE_levels = Energy_diff(nu_i,Li,nu_f,Lf)

    a_p_i, a_o_i = alpha_para_ortho_pure(nu_i,Li)
    a_p_f, a_o_f = alpha_para_ortho_pure(nu_f,Lf)

    def eps(DM):
        if DM == 0:
            return eps_0
        elif DM == 2 or DM == -2:
            return eps_pm2
        elif DM == 1 or DM == -1:
            return eps_pm1
        else:
            return np.array([0,0,0])
    
    def n(DM):
        if DM == 0:
            return n_0
        elif DM == 2 or DM == -2:
            return n_pm2
        elif DM == 1 or DM == -1:
            return n_pm1
        else:
            return np.array([0,0,0])
    
    a_i_0 = np.diag(n(0)[2]**2 * a_p_i + (n(0)[0]**2+n(0)[1]**2) * a_o_i)
    a_f_0 = np.diag(n(0)[2]**2 * a_p_f + (n(0)[0]**2+n(0)[1]**2) * a_o_f)
    Delta_alpha_0 = np.array([[a_f_0[j] - a_i_0[i] for j in range(len(states_f))] for i in range(len(states_i))])

    a_i_pm1 = np.diag(n(1)[2]**2 * a_p_i + (n(1)[0]**2+n(1)[1]**2) * a_o_i)
    a_f_pm1 = np.diag(n(1)[2]**2 * a_p_f + (n(1)[0]**2+n(1)[1]**2) * a_o_f)
    Delta_alpha_pm1 = np.array([[a_f_pm1[j] - a_i_pm1[i] for j in range(len(states_f))] for i in range(len(states_i))])

    a_i_pm2 = np.diag(n(2)[2]**2 * a_p_i + (n(2)[0]**2+n(2)[1]**2) * a_o_i)
    a_f_pm2 = np.diag(n(2)[2]**2 * a_p_f + (n(2)[0]**2+n(2)[1]**2) * a_o_f)
    Delta_alpha_pm2 = np.array([[a_f_pm2[j] - a_i_pm2[i] for j in range(len(states_f))] for i in range(len(states_i))])


    Bsen_header = "B [T]"
    Bs_table = Bvalues
    if Bmax<=1e-3:
        Bsen_header = "B [$\mu$T]"
        Bs_table = Bvalues*1e6
    elif Bmax<=1:
        Bsen_header = "B [mT]"
        Bs_table = Bvalues*1e3

    transitions = insensitive_transitions_ranges(nu_i,Li,DEB_i,nu_f,Lf,DEB_f,Bvalues,sensitivity_threshold)

    insensitive_E2_transitions = []

    Theta_m2 = mat_Theta(-2,nu_i,Li,nu_f,Lf)
    Theta_m1 = mat_Theta(-1,nu_i,Li,nu_f,Lf)
    Theta_0 = mat_Theta(0,nu_i,Li,nu_f,Lf)
    Theta_p1 = mat_Theta(1,nu_i,Li,nu_f,Lf)
    Theta_p2 = mat_Theta(2,nu_i,Li,nu_f,Lf)
    
    def Theta(q):
        if q == -2:
            return Theta_m2
        elif q == -1:
            return Theta_m1
        elif q == 0:
            return Theta_0
        elif q == 1:
            return Theta_p1
        elif q == 2:
            return Theta_p2

    def g(q,DM):
        return np.einsum('i,ij,j->',eps(DM),c_2(q),n(DM))
    
    Tifen_pure_0 = np.sum(1/3 *Theta(q)*g(q,0) for q in [-2,-1,0,1,2])
    Tifen_pure_pm1 = np.sum(1/3 *Theta(q)*g(q,1) for q in [-2,-1,0,1,2])
    Tifen_pure_pm2 = np.sum(1/3 *Theta(q)*g(q,2) for q in [-2,-1,0,1,2])

    for transition in transitions:
        i, f, Brange, min_sensibility, B, index_B = transition
        EVs_i_B = EVs_i[:,index_B]
        EVs_f_B = EVs_f[:,index_B]

        B_table = round(Bs_table[index_B],6)

        F_i,J_i,MJ_i = states_i[i]
        F_f,J_f,MJ_f = states_f[f]
        latex_MJ_i, latex_MJ_f = to_latex_fraction(MJ_i,True), to_latex_fraction(MJ_f,True)

        q = int(MJ_f - MJ_i)
        if abs(q)<=2:
            Tqif = np.einsum('ik,kl,jl->ij',EVs_i_B,Theta(-q),EVs_f_B)[i,f]#since <i|Theta|f> = <f|Theta|i>*
            if q == 0:
                Tifen = np.einsum('ik,kl,jl->ij',EVs_i_B,Tifen_pure_0,EVs_f_B)[i,f]
                Delta_alpha = Delta_alpha_0[i,f]
            elif q==1 or q==-1:
                Tifen = np.einsum('ik,kl,jl->ij',EVs_i_B,Tifen_pure_pm1,EVs_f_B)[i,f]
                Delta_alpha = Delta_alpha_pm1[i,f]
            elif q==2 or q==-2:
                Tifen = np.einsum('ik,kl,jl->ij',EVs_i_B,Tifen_pure_pm2,EVs_f_B)[i,f]
                Delta_alpha = Delta_alpha_pm2[i,f]
        else:
            Tqif = 0
            Tifen = 0

        DE_if = round(EWs_f[f,index_B] - EWs_i[i,index_B],4)
        transition_wavelenght = c/(1e6*(DE_levels+DE_if))

        if abs(Tqif) > prob_threshold:
            shift = -m_e *transition_wavelenght**2 /(4*hbar *pulsetime**2) *Delta_alpha/abs(Tifen)**2
            if show_Tifen:
                row = [i, latex_MJ_i, f, latex_MJ_f, B_table, round(Delta_alpha,5), str(np.round(abs(Tifen)*1e4,4)) + "(-4)", DE_if, round(shift,3)]
            else:
                row = [i, latex_MJ_i, f, latex_MJ_f, B_table, round(Delta_alpha,5), q, str(round(Tqif*1e3,5)) + "(-3)", DE_if, round(shift,3)]
        else:
            shift = "-"
            if show_Tifen:
                row = [i, latex_MJ_i, f, latex_MJ_f, B_table, round(Delta_alpha,5), 0, DE_if, shift]
            else:
                row = [i, latex_MJ_i, f, latex_MJ_f, B_table, round(Delta_alpha,5), q, 0, DE_if, shift]
        insensitive_E2_transitions.append(row)
    
    
    if show_Tifen:
        headers1 = ["\multicolumn{2}{c|}{$(\\nu, L)="+f"({nu_i},{Li})"+"$}", "\multicolumn{2}{c|}{$(\\nu, L)="+f"({nu_f},{Lf})"+"$}", "", "", "", "\multicolumn{2}{c}{" + f"$f_0 = {DE_levels}$ MHz"+ "}"]
        headers2 = ["$i$", "$M_J$", "$f$", "$M_J'$", "B [T]", "$\Delta \\alpha_{if}$", "$|\Theta_{\epsilon,n}^{if}|$", "$\Delta f_{hf}$ [MHz]", "$\Delta f_{AC}$ [Hz]"]
        shape = "cc|cc|c|c|c|cc"
    else:
        headers1 = ["\multicolumn{2}{c|}{$(\\nu, L)="+f"({nu_i},{Li})"+"$}", "\multicolumn{2}{c|}{$(\\nu, L)="+f"({nu_f},{Lf})"+"$}", "", "", "", "", "\multicolumn{2}{c}{" + f"$f_0 = {DE_levels}$ MHz"+ "}"]
        headers2 = ["$i$", "$M_J$", "$f$", "$M_J'$", "B [T]", "$\Delta \\alpha_{if}$", "$q$", "$\Theta_{q}^{if}$", "$\Delta f_{hf}$ [MHz]", "$\Delta f_{AC}$ [Hz]"]
        shape = "cc|cc|c|c|cc|cc"

    latex_table = "\\begin{longtable}{" + shape + "}\n" + " & ".join(headers1) + "\\\\ \n" + " & ".join(headers2) + "\\\\ \hline\n"

    for row in insensitive_E2_transitions:
        latex_table += " & ".join(str(x) for x in row) + "\\\\ \n"
    latex_table += "\\end{longtable}"
    print(latex_table)


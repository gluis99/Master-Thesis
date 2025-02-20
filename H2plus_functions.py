from H2plus_lib import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ipywidgets import interact, interactive, Dropdown, FloatLogSlider, FloatSlider, IntText, Checkbox
import fractions

#sets a color for each different set of states (F,J) that is displayed in the plots given the L level
def lev_col(L,F,J):
    """
    Sets a color for each different set of states (F, J) that is displayed in the plots.

    Parameters
    ----------
    F : float
        The F state.
    J : float
        The J state.

    Returns
    -------
    str
        The color corresponding to the given set of states.

    Raises
    ------
    ValueError
        If the provided values of F and J do not correspond to any known set of states.
    """
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

#Computes the Hamiltonian for a rovibrational level (nu,L) of H2+ given the hyperfine Hamiltonian, the Zeeeman Hamiltonian and the magnetic field
def H_nuL(Hhf,Hz,B):
    return Hhf+B*Hz

############################################################################################################
#Computes dE/dB at a certain B0
def dE_B(nu,L,B0,display=False,delta=1e-7):
    """
    Computes dE/dB at a certain B0. This function calculates the derivative of the eigenvalues of the Hamiltonian matrix H with respect to the magnetic field B at a certain B0.
    It uses a finite difference method with a small delta to calculate the derivative.

    Parameters
    ----------
    nu : int
        The vibrational level.
    L : int
        The rotational level.
    B0 : float
        The B value at which to calculate the derivative.
    display : bool, optional
        If True, the function prints the derivative for each state. Default is False.
    delta : float, optional
        The small change in B used for the finite difference calculation. Default is 1e-10.

    Returns
    -------
    dE : ndarray
        The derivative of the eigenvalues with respect to B.
    """
    H = lambda B: H_hf(nu,L)+B*normHz(nu,L)
    statesFJM = states_FJM(L)
    N = len(statesFJM)
    
    Bp = B0+delta/2
    Bm = B0-delta/2
    
    Em, EVm = get_EWs_EVs_B0(nu,L,Bm)
    Ep, EVp = get_EWs_EVs_B0(nu,L,Bp)

    dE  = (Ep-Em)/delta
    
    if display:
        for i in range(Nred):
            F,J = statesFJ[i]
            
            for k in range(int(2*J+1)):
                MJ = k-J
                
                print ("++++++++++++++++++++++++++++++++++++++++++++++")
                print('F = {0}, J = {1}, MJ = {2}'.format(F,J,MJ))
                print('dE/dB = {0}'.format(dE[states_FJM_index(F, J, MJ, L)]))
    return dE

def dE_B_5points(nu,L,B0,delta=1e-4):
    """
    Computes dE/dB at a certain B0 using a 5-point derivative. This function calculates the derivative of the eigenvalues of the Hamiltonian matrix H with respect to the magnetic field B at a certain B0.
    It uses a 5-point derivative method with a small delta to calculate the derivative.

    Parameters
    ----------
    nu : int    
        The vibrational level.
    L : int
        The rotational level.
    B0 : float
        The B value at which to calculate the derivative.
    display : bool, optional
        If True, the function prints the derivative for each state. Default is False.
    delta : float, optional
        The small change in B used for the finite difference calculation. Default is 1e-10.

    Returns
    -------
    dE : ndarray
        The derivative of the eigenvalues with respect to B.
    """
    H = lambda B: H_hf(nu,L)+B*normHz(nu,L)
    
    statesFJ = states_FJ(L)
    statesFJM = states_FJM(L)
    Nred = len(statesFJ)
    N = len(statesFJM)
    
    EWs_Bm2 = np.linalg.eigvalsh(H(B0-2*delta))
    EWs_Bm1 = np.linalg.eigvalsh(H(B0-delta))
    EWs_Bp1 = np.linalg.eigvalsh(H(B0+delta))
    EWs_Bp2 = np.linalg.eigvalsh(H(B0+2*delta))

    dE = (EWs_Bm2-8*EWs_Bm1+8*EWs_Bp1-EWs_Bp2)/(12*delta)

def dE_B_expensive(nu,L,B0):
    Bs = np.linspace(B0-1e-6,B0+1e-6,1000)
    EWs, EVs, dE_Bs = get_EWs_EVs_derivatives(nu, L, Bs)
    dE_B = dE_Bs[:,500]
    return dE_B

def dEs(nu,L,Bvalues):
    Hhf = H_hf(nu, L)
    Hz = normHz(nu, L)
    H = lambda B_: Hhf + B_*Hz
    EWs = []
    for B in Bvalues:
        EWs_B = np.linalg.eigvalsh(H(B))
        EWs.append(EWs_B)
    eigenvalues = np.array(EWs)
    dE = np.gradient(eigenvalues, Bvalues, axis=0)
    return dE
        
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

def find_zero_sensibility_B(sensibilities, Bvalues, threshold=1):
    B_values_zero_sensibility = []
    for k, sensibilities_B in enumerate(sensibilities):
        B_k = Bvalues[k]
        for l, (i, f, sensibility_if) in enumerate(sensibilities_B):
            if abs(sensibility_if) < threshold:
                B_values_zero_sensibility.append((Bvalues[k], i, f, sensibility_if))
    return B_values_zero_sensibility

def find_sign_change_B(sensibilities, Bvalues):
    previous_sensibility_if = None
    B_values_sign_change = []
    for k, sensibilities_B in enumerate(sensibilities):
        for l, (i, f, sensibility_if) in enumerate(sensibilities_B):
            if previous_sensibility_if is not None and np.sign(sensibility_if) != np.sign(previous_sensibility_if):
                B_values_sign_change.append(Bvalues[k])
            previous_sensibility_if = sensibility_if
    return B_values_sign_change

############################################################################################################
#Table of hyperfine transitions for a given (nu,L) level of H2+ and a given magnetic field value B
#Shows Energy difference, magnetic field sensitivity, transition probabilities
#Only those transitions are shown that have a probability above a certain threshold
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
    ## Get the hyperfine Hamiltonian
    #Hhf = H_hf(nu,L)
    #Hz = normHz(nu,L)
    #H = Hhf+B*Hz
    ## Get the eigenvalues and eigenvectors of the Hamiltonian for the given B
    #EWs, EVs = np.linalg.eigh(H)
    ## Get derivative of the eigenvalues with respect to B
    #dE_B_ = dE_B(nu,L,B)

    # Get the eigenvalues, eigenvectors and derivatives for the given B
    EWs_B, EVs_B = get_EWs_EVs_B0(nu, L, B)
    dE_B_ = dE_B(nu,L,B)

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
    EWsi_B, EVsi_B = get_EWs_EVs_B0(nu_i, Li, B)
    dEi_B_ = dE_B(nu_i,Li,B)
    EWsf_B, EVsf_B = get_EWs_EVs_B0(nu_f, Lf, B)
    dEf_B_ = dE_B(nu_f,Lf,B)
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
#Table of electric quadrupole transitions for a given (nu_i,Li) to (nu_f,Lf) level of H2+

def n_xz_angle(beta):
    return np.array([np.sin(beta),0,np.cos(beta)])
#gives polarizationvector for a given propagation direction n and a given polarization q= -1(right), 0(linear), 1(left)
def epsilon_polarization(n,q):
    if q == -1:
        return np.array([0,1j,0])
    elif q == 0:
        return n
    elif q == 1:
        return np.array([0,-1j,0])

#For a given magnetic field value B and angle beta between propagation vector and quantization axis, for linear and left circular polarization
def table_quadrupole_transitions(nu_i,Li,nu_f,Lf,B,beta=0,prob_threshold=1e-3, sensitivity_threshold=None):
    # Get the eigenvalues, eigenvectors and derivatives for the given B
    EWsi_B, EVsi_B = get_EWs_EVs_B0(nu_i, Li, B)
    dEi_B_ = dE_B(nu_i,Li,B)
    EWsf_B, EVsf_B = get_EWs_EVs_B0(nu_f, Lf, B)
    dEf_B_ = dE_B(nu_f,Lf,B)
    # Get the number of states
    Ni = len(EWsi_B)
    Nf = len(EWsf_B)
    if redTheta(nu_i,Li,nu_f,Lf) == None:
        print('No electric quadrupole transitions for the given levels implemented.')
        return
    
    n = n_xz_angle(beta)

    # Initialize empty lists to store the transition data
    transitions = []
    for i in range(Ni):
        for j in range(Nf):
            # Calculate the energy difference
            diff_Eij = EWsf_B[j]-EWsi_B[i]
            # Calculate the magnetic field sensitivity
            diff_dE_Bij = dEf_B_[j]-dEi_B_[i]
            # If the transition probability is above the threshold, add the transition data to the list
            if prob_threshold is None or abs( quadrupole_operator(nu_i,Li,nu_f,Lf,epsilon_polarization(n,0),n)[i,j]) > prob_threshold \
                or abs(quadrupole_operator(nu_i,Li,nu_f,Lf,epsilon_polarization(n,1),n)[i,j]) > prob_threshold:
                if sensitivity_threshold is None or abs(diff_dE_Bij) < sensitivity_threshold:
                    transitions.append([i, j, diff_Eij, diff_dE_Bij, quadrupole_operator(nu_i,Li,nu_f,Lf,epsilon_polarization(n,0),n)[i,j], quadrupole_operator(nu_i,Li,nu_f,Lf,epsilon_polarization(n,1),n)[i,j]])
    if len(transitions) == 0:
        print('No transitions with the given thresholds.')
        return
    # Create a DataFrame from the list of transitions
    df = pd.DataFrame(transitions, columns=['Initial State', 'Final State', 'Transition frequency [MHz]', \
                            'Magnetic Field Sensitivity [MHz/T]', '$Linear pol. at \\beta$', '$Left circular pol. at \\beta$'])
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
    frac = fractions.Fraction(num).limit_denominator()
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
        color_legend = {lev_col(F, J): f'({F}, {J})' for F, J, _ in states}
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

# Function to update the plots for a fixed B field value   
def update_plots(B, scale):
    #Calculate the eigenvalues and eigenvectors for the given B field
    EWs_atB, EVs_atB = np.linalg.eigh(H(B))
    
    display(Markdown(f'The indices of the eigenstates are ordered in ascending order of their energy.'))

    # Plot the eigenvectors
    figs_axes_B = plot_eigenvectors(L, range(N), EWs_atB, EVs_atB, B, scale)
    for fig, ax in figs_axes_B:
        display(fig)

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
#insensitive transition (0,1) at B = 4.582582582583e-4 T for F,J,MJ: (1/2,3/2,1/2) <-> (3/2, 5/2, 1/2)

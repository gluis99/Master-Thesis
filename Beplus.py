#library for Be+ ion
import numpy as np
import math
import matplotlib.pyplot as plt
from sympy.physics.wigner import wigner_3j as w3j
from sympy.physics.wigner import wigner_6j as w6j
from sympy.physics.quantum.cg import CG
from scipy.linalg import eig
from scipy.sparse.linalg import eigs
from scipy import constants
from math import pi
from fractions import Fraction

#contants

g_L = 1 #Landé g-factor
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
m_e = constants.m_e
m_p = constants.m_p

S_e = 0.5

I = 3/2

########################################################################################################################

#g-Factor values from Shiga et al ., Phys . Rev . A 84, 012510 (2011)
g_J_ges = 2.00226239
g_ratio = 2.1347798527e-4
gt_I = g_ratio *g_J_ges

def g_J(L, J, S=0.5, gL=g_L, ge=g_e):
    if L == 0 and J == 0.5:
        gJv = g_J_ges
    else:
        gJv = gL * (J * (J + 1) - S * (S + 1) + L * (L + 1)) / (2 * J * (J + 1)) + \
              ge * (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))
    return  gJv

#Hyperfine constants for (0,1/2) from Shiga et al ., Phys . Rev . A 84, 012510 (2011) 
# the others from Puchalski and Pachucki, Phys . Rev . A 032510 (2009)
def Ahf(L,J,B=0):
    #diamagnetic shift coefficient from Shiga et al ., Phys . Rev . A 84, 012510 (2011)
    k = 2.63e-11
    correction = 1+k*B**2
    As = {(0, 1/2): -625008837.044,
            (1, 1/2): -117932e3,
            (1, 3/2): -1026e3}
    if (L,J) in As:
        return As[(L, J)]*correction
    else:
        print(f"Ahf({L},{J}) not available.")
        return None

def Bhf(L,J):
    data = {(0, 1/2): 0,
            (1, 1/2): 0,
            (1, 3/2): -2.29940e6}
    if (L,J) in data:
        return data[(L, J)]
    else:
        print(f"Bhf({L},{J}) not available.")
        return None

########################################################################################################################
# Hyperfine states
# J = S_e + L, F = J + I
def states_Fm(J):
    #returns the states of the fine structure multiplet
    states = []
    Fs = np.arange(abs(J-I),J+I+1, dtype=Fraction)
    for F in Fs:
        nm = int(2*F+1)
        for k in range(nm):
            mF = -F+k
            states.append([F,mF])
    states = np.reshape(states, (-1,2))
    return states

def states_F(J):
    #returns the states of the fine structure multiplet
    states = []
    Fs = np.arange(abs(J-I),J+I+1, dtype=Fraction)
    for F in Fs:
        states.append(F)
    return states

def states_Fm_index(F,mF, J):
    """
    Returns the index of the state with quantum numbers F and J in the array of states with
    orbital angular momentum quantum number L.

    Parameters
    ----------
    F : int
        
    mF : int
        

    Returns
    -------
    int
        The index of the state with quantum numbers F and mF in the array of states with
        J.
    """
    states = states_Fm(J)
    for i in range(len(states)):
        if states[i][0] == F and states[i][1] ==mF:
            return i
    return None


########################################################################################################################
# Hamiltonians in MHz

def Iq(J,q):
    states = states_Fm(J)
    N = len(states)
    MIq = np.zeros((N,N))

    for i in range(N):
        for f in range(N):
            F,mF = states[i]
            Fp,mFp = states[f]

            WE = CG(Fp,mFp,1,q,F,mF).doit()
            redI = (-1)**(F+J+I+1) *w6j(I,1,I,Fp,J,F) *np.sqrt(2*Fp+1) *np.sqrt(I*(I+1)*(2*I+1))
            MIq[i,f] = WE*redI
    return MIq

def Jq(J,q):
    states = states_Fm(J)
    N = len(states)
    MJq = np.zeros((N,N))

    for i in range(N):
        for f in range(N):
            F,mF = states[i]
            Fp,mFp = states[f]

            WE = CG(Fp,mFp,1,q,F,mF).doit()
            redJ = (-1)**(Fp+J+I+1) *w6j(J,1,J,Fp,I,F) *np.sqrt(2*Fp+1) *np.sqrt(J*(J+1)*(2*J+1))
            MJq[i,f] = WE*redJ
    return MJq

def H_hf(L,J,B=0):
    #A = Ahf(L,J)
    #B = Bhf(L,J)

    states = states_Fm(J)
    N = len(states)

    IJ = np.zeros((N,N))
    Quadrupole = np.zeros((N,N))
    for i in range(N):
        for f in range(N):
            F,mF = states[i]
            Fp,mFp = states[f]

            if F == Fp:
                if mF == mFp:
                    IJ[i,f] = (F*(F+1)-I*(I+1)-J*(J+1))/2
    if J != 1/2:
        Quadrupole = (3*IJ*(IJ) + 3/2*(IJ) - I*(I+1)*J*(J+1)*np.eye(N))/(2*I*(2*I-1)*J*(2*J-1))
    
    return (Ahf(L,J,B)*IJ + Bhf(L,J)*Quadrupole)*1e-6

def mu(L,J,q):
    return - gt_I*µ_b*Iq(J,q) - g_J(L, J)*µ_b*Jq(J,q)

def H_z(L,J,B):
    return - mu(L,J,0)*B*1e-6

def Htot(L,J,B):
    return H_z(L,J,B) + H_hf(L,J,B)

def H_J(Hhf,Hz,B):
    return Hz*B + Hhf

########################################################################################################################
#Eigenvalues and eigenvectors of the Hamiltonian at a given magnetic field B

#returns arrays of the EWs and the EVs corresponding to them ordered in descending energy order
def ES_atB(L,J,B):
    H = Htot(L,J,B)
    EWs,EVs = np.linalg.eigh(H)
    return EWs,EVs

#orders the eigenvalues to match the previous ones
#problem for very small field, exchanges degenerate states
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

#For a given electric angular momentum (J), calculates the eigenvalues and eigenvectors
#of the total Hamiltonian for a given magnetic field B0
def get_EWs_EVs_B0(L,J,B0):
    H = Htot(L,J,B0)
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

########################################################################################################################

#For a given J, calculates the eigenvalues, eigenvectors and derivatives
#of the total Hamiltonian for a given set of magnetic field values Bvalues
def get_EWs_EVs_derivatives(L,J,Bvalues):
    """
    Calculates the eigenvalues, eigenvectors and derivatives of the Hamiltonian for a given L and nu level.

    Parameters
    ----------
    L : int
        The orbital angular momentum quantum number.

    J : float
      
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
    #Get the hyperfine and zeeman Hamiltonians
    #the hyperfine constant Ahf is taken to be that by B=0
    matrix_Hhf = H_hf(L,J,0)
    matrix_Hz = H_z(L,J,1)

    N = len(matrix_Hhf)

    # Initialize empty lists to store all the eigenvalues and eigenvectors for each B
    EWs = []
    EVs_all = []
    for i in range(N):
        EWs.append([])
        EVs_all.append([])

    EVs_prev = np.eye(N)

    # Loop through each value of B
    for Bi in Bvalues:
        # Calculate the hamiltonian matrix
        Htot = H_J(matrix_Hhf,matrix_Hz,Bi)
    
        # Calculate the eigenvalues of H using np.linalg.eigh potentially in the wrong order
        EWs_wo_B,EVs_wo_B = np.linalg.eigh(Htot)
    
        EWs_B = np.zeros(N)
        EVs_B = np.zeros((N,N))
    
        #limiting B field to change approach of ordering
        #chosen not too small, but small enough for all implemented (nu,L)
        Bchange=1e-9
    
        #get the eigenvalues in the right order
        #naive approach, assuming the mixing is small (for L=1it is for B<588µT)
        #only consider biggest entry of EV to order (compare EVs to identity matrix)
        if Bi<=Bchange:
            for i in range(N):
                i_ordered = np.argmax(abs(EVs_wo_B[:,i]))
                EWs_B[i_ordered] = EWs_wo_B[i]
                EVs_B[:,i_ordered] = EVs_wo_B[:,i]
            #assure all eigenvalues are taken into account
            #print if there is a EWs_B[i] that is =0 and there are no EWs_wo_B[i] =0
            for i in range(N):
                if EWs_B[i]==0:
                    for j in range(N):
                        if EWs_wo_B[j]==0:
                            break
                    print("EWs_B[i] = 0 and EWs_wo_B[j] != 0 so something went wrong")
    
        #When the degeneracy in MJ is lifted enough this approach is more exact
        #Compare EVs at B to the EVs with the previous B
        else:
            EWs_B, EVs_B = order_states_wprev(EVs_prev,EWs_wo_B,EVs_wo_B)
            EVs_prev = EVs_B
    
        #append EVs at B to the list of all eigenvalues and eigenvectors
        for i in range(N):
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

########################################################################################################################

#sets a color for each different set of states (F) that is displayed in the plots given the L level
def lev_col(F):
    """
    Sets a color for each different set of states (F) that is displayed in the plots.

    Parameters
    ----------
    F : float
        The F state.
        

    Returns
    -------
    str
        The color corresponding to the given set of states.

    Raises
    ------
    ValueError
        If the provided values of F and J do not correspond to any known set of states.
    """
    if F==0:
        return 'b'
    elif F==1:
        return 'r'
    elif F==2:
        return 'g'
    elif F==3:
        return 'y'
    elif F==4:
        return 'm'
    elif F==5:
        return 'c'
    else:
        raise ValueError("Invalid values of F and J.")


def plot_eigenenergies(L,J, B_max, number_points_B=1000, separate=True, show_derivatives=True):
    #Get the eigenvalues, eigenvectors and derivatives
    B_values = np.linspace(0, B_max, number_points_B)
    eigenvalues_, eigenvectors_, derivatives_ = get_EWs_EVs_derivatives(L,J, B_values)
    
    statesF = states_F(J)
    statesFm = states_Fm(J)

    #Get the number of states
    Nred = len(statesF)
    N = len(statesFm)

    ptit = 'Eigenenergies for $\\mathbf{{L}},\\mathbf{{J}}$=({0},{1})'.format(int(L),Fraction(J).limit_denominator())

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

    # Plot the eigenvalues as a function of B for the different F,J
    for i in range(Nred):
        if separate or i == 0:  # Create a new figure for the first pair or if Separate is checked
            if show_derivatives:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            else:
                fig, ax1 = plt.subplots(figsize=(6, 6))

        F = statesF[i]
        col = lev_col(F)
        label_shown = False
        for k in range(int(2*F+1)):
            mF = k-F
            energy = eigenvalues_[states_Fm_index(F, mF, J)]
            ax1.plot(Bplot, energy, color=col)
            if not label_shown:
                ax1.plot([],[],color=col, label='F={}'.format(int(F)))
                label_shown = True

            # Plot the derivative of the energy if the Derivatives checkbox is checked
            if show_derivatives:
                derivative = derivatives_[states_Fm_index(F, mF, J)]
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

########################################################################################################################
from IPython.display import display, Markdown
import pandas as pd
from ipywidgets import interact, interactive, Dropdown, FloatLogSlider, FloatSlider, IntText, Checkbox

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
def to_latex_fraction(num):
    frac = Fraction(num).limit_denominator()
    if frac < 0:
        return r'$-\frac{{{}}}{{{}}}$'.format(abs(frac.numerator), frac.denominator)
    else:
        return r'$\frac{{{}}}{{{}}}$'.format(frac.numerator, frac.denominator) if frac.denominator != 1 else str(frac.numerator)

#table explaining content of plot
def display_table(J):
    states_ = states_Fm(J)
    N_ = len(states_)
    numberofstate = range(N_)
    colors = [lev_col(states_[j][0]) for j in numberofstate]

    #Explaining content of table
    display(Markdown(f'In each plot a column represents the coefficient squared of the eigenstate\
                      corresponding to the basis state $|F, m_F \\rangle$.\
                     The color of the column corresponds to the $F$ state of the bar.'))

    # Create a DataFrame for the table
    pd.set_option('display.html.use_mathjax', True)
    data = {
        'Column': [i for i in numberofstate],
        '$F$': [to_latex_fraction(states_[i][0]) for i in numberofstate],
        '$m_F$': [to_latex_fraction(states_[i][1]) for i in numberofstate],
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

def table_eigenstates(L,J,B):
    states_ = states_Fm(J)
    N_ = len(states_)
    numberofstate = range(N_)
    colors = [lev_col(states_[j][0]) for j in numberofstate]

    EWs, EVs = get_EWs_EVs_B0(L, J, B)

    #Explaining content of table
    display(Markdown(f'In each plot a column represents the coefficient squared of the eigenstate\
                      corresponding to the basis state $|F, m_F \\rangle$.\
                     The color of the column corresponds to the $F$ state of the bar.'))

    # Create a DataFrame for the table
    pd.set_option('display.html.use_mathjax', True)
    data = {
        'Column': [i for i in numberofstate],
        '$F$': [to_latex_fraction(states_[i][0]) for i in numberofstate],
        '$m_F$': [to_latex_fraction(states_[i][1]) for i in numberofstate],
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



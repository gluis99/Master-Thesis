from H2plus_lib import *
from H2plus_functions import insensitive_transitions, insensitive_transitions_same, to_latex_fraction, insensitive_transitions_ranges
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ipywidgets import interact, interactive, Dropdown, FloatLogSlider, FloatSlider, IntText, Checkbox
import fractions
from tabulate import tabulate

def get_hf_indices_latex(L):
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

def get_hf_indices_energies_B_latex(nu,L,B):
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

def get_insensitive_transitions_latex_same(nu,L,Bmin,Bmax,points=1000, sensitivity=1):
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

    headers = ["State index", "($F,J,M_J$)", "State index", "($F,J,M_J$)", "Frequency [MHz]", B_header]
    latex_table = "\\begin{tabular}{cc|cc|c|c}\n" + " & ".join(headers) + "\\\\ \hline\n"
    for row in table:
        latex_table += " & ".join(str(x) for x in row) + "\\\\ \n"
    latex_table += "\\end{tabular}"
    print(latex_table)

def get_insensitive_transitions_latex(nu_i,Li,nu_f,Lf,Bmin,Bmax,points=1000, sensitivity=1,with_sensitivity=False,inlatextable=False):
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
        headers2 = ["State index", "($F,J,M_J$)", "State index", "($F,J,M_J$)", "Frequency [MHz]", B_header, "Sensitivity [MHz/T]"]
        shape = "cc|cc|c|c|c"
    else:
        headers2 = ["State index", "($F,J,M_J$)", "State index", "($F,J,M_J$)", "Frequency shift [MHz]", B_header]
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

def get_insensitive_transitions_ranges_latex(nu_i,Li,nu_f,Lf,Bmin,Bmax,points=1000, sensitivity=1,with_sensitivity=False,inlatextable=False):
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
        headers2 = ["State index", "\\multicolumn{3}{c|}{($F,J,M_J$)}", "State index", "\\multicolumn{3}{c|}{($F,J,M_J$)}", "$\\Delta f_{{hf}}$ [MHz]", Bmin_header, Bmax_header, Bsen_header, "Min sens. [MHz/T]"]
        shape = "c|ccc|c|ccc|c|cc|cc"
    else:
        headers2 = ["State index", "\\multicolumn{3}{($F,J,M_J$)}", "State index", "\\multicolumn{3}{($F,J,M_J$)}", "$\\Delta f_{{hf}}$ [MHz]", Bmin_header, Bmax_header]
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

def get_insensitive_mag_dipole_transitions_latex(nu,L,Bmin,Bmax,samples=1000,prob_threshold=1e-5, sensitivity_threshold=10,inlatextable=False):
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

        if prob_threshold is None or abs(mu_p[i,f]) > prob_threshold or abs(mu_0[i,f]) > prob_threshold or abs(mu_m[i,f]) > prob_threshold:
            insensitive_M1_transitions.append([i, f, DE_if, Bsen, round(mu_p[i,f],5), round(mu_0[i,f],5), round(mu_m[i,f],5)])
        
    headers1 = [f"$(\\nu, L)=({nu},{L})$", "", "", "", "", ""]
    headers2 = ["State index", "State index", "Frequency shift [MHz]", Bsen_header, "$\mu_+$", "$\mu_0$", "$\mu_-$"]
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

def get_insensitive_quadrupole_transitions_onetheta_latex(nu_i,Li,nu_f,Lf,Bmin,Bmax,samples=1001,prob_threshold=1e-5, sensitivity_threshold=10,inlatextable=False):
    
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
            Tqif = np.einsum('ik,kl,jl->ij',EVs_i_B,Theta(q),EVs_f_B)[i,f]
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

def get_insensitive_twophoton_transitions_latex(nu_i,Li,nu_f,Lf,Bmin,Bmax,samples=1001,prob_threshold=1e-5, sensitivity_threshold=10,inlatextable=False):
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
        q = int(MJ_i - MJ_f)
        if SQ(q) is not None:
            S_qif = np.einsum('ik,kl,jl->ij',EVs_i_B,SQ(q),EVs_f_B)[i,f]
        else:
            S_qif = 0

        DE_if = round(EWs_f[f,index_B] - EWs_i[i,index_B],4)

        if prob_threshold is None or abs(S_qif) > prob_threshold:
            insensitive_2E1_transitions.append([i, latex_MJ_i, f, latex_MJ_f, DE_if, B_table, q, round(S_qif,5)])
        
    headers1 = [f"$(\\nu_i, L_i)=({nu_i},{Li})$", "", f"$(\\nu_f, L_f)=({nu_f},{Lf})$", "", f"$\\Delta E = {DE_levels}$ THz", "", "", ""]
    headers2 = ["i", "$M_J$", "f", "$M_J'$", "Frequency shift [MHz]", Bsen_header, "$q$", "$^SQ_{qq}$"]
    shape = "cc|cc|c|c|cc"

    latex_table = ""

    if inlatextable:
        latex_table += "\\begin{table}[ht]\n"
        latex_table += "\\centering\n"
        latex_table += "\\resizebox{\\textwidth}{!}{%\n"

    latex_table += "\\begin{tabular}{" + shape + "}\n" + " & ".join(headers1) + "\\\\ \n" + " & ".join(headers2) + "\\\\ \hline\n"
    for row in insensitive_2E1_transitions:
        latex_table += " & ".join(str(x) for x in row) + "\\\\ \n"
    latex_table += "\\end{tabular}"

    if inlatextable:
        latex_table += "}\n"
        latex_table += "\\caption{Caption}\n"
        latex_table += "\\label{tab:my_label}\n"
        latex_table += "\\end{table}"

    print(latex_table)

def get_insensitive_quadrupole_transitions_latex(nu_i,Li,nu_f,Lf,Bmin,Bmax,points=1000, sensitivity=1):
    Bvalues = np.linspace(Bmin,Bmax,points)
    transitions = insensitive_transitions(nu_i,Li,nu_f,Lf,Bvalues,sensitivity)# transitions is a list of tuples (B, i, f, sensitivity)
    DE_levels = Energy_diff(nu_i,Li,nu_f,Lf)
    table = []

    Theta_0 = mat_Theta(0,nu_i,Li,nu_f,Lf)
    Theta_1 = mat_Theta(1,nu_i,Li,nu_f,Lf)
    Theta_2 = mat_Theta(2,nu_i,Li,nu_f,Lf)

    B_header = "B [T]"
    if Bmax<=1e-3:
        B_header = "B [$\mu$T]"
    elif Bmax<=1:
        B_header = "B [mT]"

    for transition in transitions:
        B, i, f, s = transition


        EB_init,EVs_i = get_EWs_EVs_B0(nu_i, Li, B)
        EB_final,EVs_f = get_EWs_EVs_B0(nu_f, Lf, B)
        DE = EB_final[f] - EB_init[i]
        DE = "{:.5f}".format(DE)  # Format DE to have 5 digits after the decimal point

        T0if = np.einsum('ik,ij,jl->kl',EVs_i,Theta_0,EVs_f)[i,f]
        T1if = np.einsum('ik,ij,jl->kl',EVs_i,Theta_1,EVs_f)[i,f]
        T2if = np.einsum('ik,ij,jl->kl',EVs_i,Theta_2,EVs_f)[i,f]

        if Bmax<=1e-3:
            B_table = round(B*1e6,4)
        elif Bmax<=1:
            B_table = round(B*1e3,8)
        else:
            B_table = round(B,12)
        row = [i,  f, DE, B_table, T0if, T1if, T2if]
        table.append(row)
    
    headers1 = ["$(\\nu_i, L_i)$=({},{})".format(nu_i, Li),"$(\\nu_f, L_f)$=({},{})".format(nu_f, Lf),"$\Delta E$ = {}".format(DE_levels)]
    headers2 = ["State index", "State index", "Frequency shift [MHz]", B_header, "$\Theta^{(2)}_0$", "$\Theta^{(2)}_1$", "$\Theta^{(2)}_2$"]
    latex_table = "\\begin{tabular}{c|c|c|c|ccc}\n" + " & ".join(headers1) + "\\\\ \n" + " & ".join(headers2) + "\\\\ \hline\n"
    for row in table:
        latex_table += " & ".join(str(x) for x in row) + "\\\\ \n"
    latex_table += "\\end{tabular}"
    print(latex_table)

############################################################################################################

def get_AC_stark_shift_twophoton_latex(nu_i,Li,nu_f,Lf,Bmin,Bmax,samples,theta=0,pulsetime=0.1,sensitivity_threshold=10):
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
        q = int(MJ_i - MJ_f)
        if SQ(q) is not None:
            S_qif = np.einsum('ik,kl,jl->ij',EVs_i_B,SQ(q),EVs_f_B)[i,f]
            atheta_i_B = np.einsum('ik,kl,jl->ij',EVs_i_B,atheta_i,EVs_i_B)[i,i]
            atheta_f_B = np.einsum('ik,kl,jl->ij',EVs_f_B,atheta_f,EVs_f_B)[f,f]
            delta_alpha = atheta_f_B - atheta_i_B
        else:
            S_qif = 0

        DE_if = round(EWs_f[f,index_B] - EWs_i[i,index_B],4)

        if abs(S_qif) > 1e-5:
            shift = -1/(8*pulsetime) * delta_alpha/abs(S_qif)
            row = [i, round(atheta_i_B,4), f, round(atheta_f_B,4), round(delta_alpha,4), round(abs(S_qif),5), DE_if/2, round(shift,3)]
        else:
            shift = "-"
            row = [i, round(atheta_i_B,4), f, round(atheta_f_B,4), round(delta_alpha,4), round(abs(S_qif),5), DE_if/2, shift]
        insensitive_2E1_transitions.append(row)

    headers1 = [f"$(\\nu_i, L_i)=({nu_i},{Li})$", "", f"$(\\nu_f, L_f)=({nu_f},{Lf})$", "", "", "", f"$f_0 = {DE_levels/2}$ MHz", ""]
    headers2 = ["State index", "$\\alpha_i$", "State index", "$\\alpha_f$", "$\Delta \\alpha$", "$|^SQ_{qq}|$", "\Delta f_{hf} [MHz]", "$\Delta f_{AC}$ [Hz]"]
    shape = "cc|cc|cc|cc"

    latex_table = "\\begin{tabular}{" + shape + "}\n" + " & ".join(headers1) + "\\\\ \n" + " & ".join(headers2) + "\\\\ \hline\n"
    for row in insensitive_2E1_transitions:
        latex_table += " & ".join(str(x) for x in row) + "\\\\ \n"
    latex_table += "\\end{tabular}"
    print(latex_table)

def get_AC_stark_shift_stretched_quadrupole_latex(even,B,pulsetime=0.1):
    nu_L_if_pairs_list_even = [((0,0),(0,2)),((0,0),(1,2)),((0,0),(2,2)),((0,0),(3,2))]
    nu_L_if_pairs_list_odd = [((0,1),(1,1)),((0,1),(3,1)),((0,1),(0,3)),((0,1),(1,3)),((0,1),(3,3))]
    if even:
        nu_L_if_pairs_list = nu_L_if_pairs_list_even
    else:
        nu_L_if_pairs_list = nu_L_if_pairs_list_odd
    
    table = []
    for nu_L_if_pair in nu_L_if_pairs_list:
        nu_i, Li = nu_L_if_pair[0]
        nu_f, Lf = nu_L_if_pair[1]
        
        dL = abs(Li - Lf)
        if dL == 0:
            n = n_0
            eps = eps_0
        elif dL == 2:
            n = n_pm2
            eps = eps_pm2

        if Li%2 == 1:
            F_i,J_i,MJ_i = 3/2, Li + 3/2, +(Li + 3/2)
            F_f,J_f,MJ_f = 3/2, Lf + 3/2, +(Lf + 3/2)
        else:
            F_i,J_i,MJ_i = 1/2, Li + 1/2, +(Li + 1/2)
            F_f,J_f,MJ_f = 1/2, Lf + 1/2, +(Lf + 1/2)
    
        index_i = states_FJM_index(F_i,J_i,MJ_i,Li)
        index_f = states_FJM_index(F_f,J_f,MJ_f,Lf)

        EB_i,EVs_i = get_EWs_EVs_B0(nu_i, Li, B)
        EB_f,EVs_f = get_EWs_EVs_B0(nu_f, Lf, B)
        DE = EB_f[index_f] - EB_i[index_i]

        a_p_i, a_o_i = alpha_para_ortho_pure(nu_i,Li)
        a_p_f, a_o_f = alpha_para_ortho_pure(nu_f,Lf)
        a_i = n[2]**2 * a_p_i[index_i,index_i] + (n[0]**2+n[1]**2) * a_o_i[index_i,index_i]
        a_f = n[2]**2 * a_p_f[index_f,index_f] + (n[0]**2+n[1]**2) * a_o_f[index_f,index_f]
        Delta_alpha = a_f - a_i

        abs_Thetaif_en = abs(quadrupole_operator_pure(nu_i,Li,nu_f,Lf,eps,n)[index_i,index_f])
        
        freq_NR = Energy_diff(nu_i, Li, nu_f, Lf, rel=False)

        freq = 1e6*mat_transition_freqs_E(nu_i, Li, EB_i, nu_f, Lf, EB_f)[index_i,index_f]
        transition_wavelenght = c/freq

        Delta_freq = -m_e *transition_wavelenght**2 /(4*hbar *pulsetime**2) *Delta_alpha/abs_Thetaif_en**2

        row = [f"({nu_i},{Li})", round(a_i,4), f"({nu_f},{Lf})", round(a_f,4), round(Delta_alpha,4), round(abs_Thetaif_en,6), round(freq_NR*1e-3,3), round(DE,3), round(Delta_freq,3)]
        table.append(row)
    
    headers1 = ["$(\\nu_i, L_i)$", "$\\alpha_i$", "$(\\nu_f, L_f)$", "$\\alpha_f$", "$\\Delta \\alpha$", "$|\Theta^{{if}}_{{\\epsilon n}}|$", "$f_{{NR}}$ [GHz]", "$\\Delta f_{{hf}}$ [MHz]", "$\Delta f_{{AC}}$ [Hz]"]
    latex_table = "\\begin{tabular}{cccc|ccccc}\n" + " & ".join(headers1) + "\\\\ \hline\n"
    for row in table:
        latex_table += " & ".join(str(x) for x in row) + "\\\\ \n"
    latex_table += "\\end{tabular}"
    print(latex_table)

def get_AC_stark_shift_quadrupole_latex(nu_i,Li,nu_f,Lf,B,pulsetime=0.1):
    states_i = states_FJM(Li)
    states_f = states_FJM(Lf)

    EWs_i, EVs_i = get_EWs_EVs_B0(nu_i, Li, B)
    EWs_f, EVs_f = get_EWs_EVs_B0(nu_f, Lf, B)
    Ni,Nf = len(EWs_i), len(EWs_f)
    DE_levels = Energy_diff(nu_i,Li,nu_f,Lf)

    a_p_i, a_o_i = alpha_para_ortho_pure(nu_i,Li)
    a_p_f, a_o_f = alpha_para_ortho_pure(nu_f,Lf)

    DL = abs(Li - Lf)
    if DL == 0:
        n = n_0
        eps = eps_0
    elif DL == 2:
        n = n_pm2
        eps = eps_pm2
    else:
        return "No quadrupole transitions possible between these rotational states"
    
    a_i = np.diag(n[2]**2 * a_p_i + (n[0]**2+n[1]**2) * a_o_i)
    a_f = np.diag(n[2]**2 * a_p_f + (n[0]**2+n[1]**2) * a_o_f)
    Delta_alpha = np.array([[a_f[j] - a_i[i] for j in range(Nf)] for i in range(Ni)])

    abs_Theta_en = np.round(abs(quadrupole_operator_pure(nu_i,Li,nu_f,Lf,eps,n)),10)

    freq_NR = Energy_diff(nu_i, Li, nu_f, Lf, rel=False)#in MHz

    freq = np.array(mat_transition_freqs_E(nu_i, Li, EWs_i, nu_f, Lf, EWs_f)) #in MHz
    transition_wavelenght = np.array([[c/(freq_NR+freq[i,j])*1e-6 for j in range(Nf)] for i in range(Ni)]) #in m
    
    table = []

    for i in range(Ni):
        for j in range(Nf):

            DE_hf = EWs_f[j] - EWs_i[i]
            
            if abs_Theta_en[i,j] == 0:
                Delta_freq = 0
            else:
                Delta_freq = -m_e *transition_wavelenght[i,j]**2 /(4*hbar *pulsetime**2) *Delta_alpha[i,j]/abs_Theta_en[i,j]**2
                row = [i, round(a_i[i],4), j, round(a_f[j],4), round(Delta_alpha[i,j],4), round(abs_Theta_en[i,j]*1e4,6), round(freq[i,j],3), round(Delta_freq,3)]
                table.append(row)

    headers1 = [f"$(\\nu_i, L_i)=({nu_i},{Li})$", "", f"$(\\nu_f, L_f)=({nu_f},{Lf})$", "", "", "",f"$f_{{NR}}={freq_NR}$ [MHz]"]
    headers2 = ["State i", "$\\alpha_i$", "State f", "$\\alpha_f$", "$\Delta \\alpha$", "$|\Theta^{{if}}_{{\\epsilon n}}|(-4)$", "$\\Delta f_{{hfs}}$ [MHz]", "$\Delta f_{{AC}}$ [Hz]"]
    latex_table = "\\begin{tabular}{cc|cc|cc|cc}\n" + " & ".join(headers1) + "\\\\ \n" + " & ".join(headers2) + "\\\\ \hline\n"
    for row in table:
        latex_table += " & ".join(str(x) for x in row) + "\\\\ \n"
    latex_table += "\\end{tabular}"

    print(latex_table)


############################################################################################################

def get_table_insensitive_quadrupole_transitions_with_shift(nu_i,Li,nu_f,Lf,Bmin,Bmax,samples=1001,prob_threshold=1e-5, sensitivity_threshold=10, pulsetime=0.1):
    Bvalues = np.linspace(Bmin, Bmax, samples)
    EWs_i, EVs_i, DEB_i = get_EWs_EVs_derivatives(nu_i,Li,Bvalues)
    states_i = states_FJM(Li)
    EWs_f, EVs_f, DEB_f = get_EWs_EVs_derivatives(nu_f,Lf,Bvalues)
    states_f = states_FJM(Lf)

    DE_levels = Energy_diff(nu_i,Li,nu_f,Lf)

    a_p_i, a_o_i = alpha_para_ortho_pure(nu_i,Li)
    a_p_f, a_o_f = alpha_para_ortho_pure(nu_f,Lf)

    DL = abs(Li - Lf)
    if DL == 0:
        n = n_0
        eps = eps_0
    elif DL == 2:
        n = n_pm2
        eps = eps_pm2
    else:
        return "No quadrupole transitions possible between these rotational states"
    
    a_i = np.diag(n[2]**2 * a_p_i + (n[0]**2+n[1]**2) * a_o_i)
    a_f = np.diag(n[2]**2 * a_p_f + (n[0]**2+n[1]**2) * a_o_f)
    Delta_alpha = np.array([[a_f[j] - a_i[i] for j in range(len(states_f))] for i in range(len(states_i))])

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
            Tqif = np.einsum('ik,kl,jl->ij',EVs_i_B,Theta(q),EVs_f_B)[i,f]
        else:
            Tqif = 0
        
        DE_if = round(EWs_f[f,index_B] - EWs_i[i,index_B],4)
        transition_wavelenght = c/(1e6*(DE_levels+DE_if))

        if abs(Tqif) > prob_threshold:
            shift = -m_e *transition_wavelenght**2 /(4*hbar *pulsetime**2) *Delta_alpha[i,f]/abs(Tqif)**2
            row = [i, latex_MJ_i, f, latex_MJ_f, B_table, round(Delta_alpha[i,f],5), q, str(round(Tqif*1e3,5)) + "(-3)", DE_if, round(shift,3)]
        else:
            shift = "-"
            row = [i, latex_MJ_i, f, latex_MJ_f, B_table, round(Delta_alpha[i,f],5), q, 0, DE_if, "-"]
        insensitive_E2_transitions.append(row)
    
    headers1 = ["\multicolumn{2}{c|}{$(\\nu, L)="+f"({nu_i},{Li})"+"$}", "\multicolumn{2}{c|}{$(\\nu, L)="+f"({nu_f},{Lf})"+"$}", "", "", "", "", "\multicolumn{2}{c}{" + f"$f_0 = {DE_levels}$ MHz"+ "}"]
    headers2 = ["$i$", "$M_J$", "$f$", "$M_J'$", "B [T]", "$\Delta \\alpha_{if}$", "$q$", "$|\Theta_{q}|$", "$\Delta f_{hf}$ [MHz]", "$\Delta f_{AC}$ [Hz]"]

    shape = "cc|cc|c|c|cc|cc"

    latex_table = "\\begin{longtable}{" + shape + "}\n" + " & ".join(headers1) + "\\\\ \n" + " & ".join(headers2) + "\\\\ \hline\n"

    for row in insensitive_E2_transitions:
        latex_table += " & ".join(str(x) for x in row) + "\\\\ \n"
    latex_table += "\\end{longtable}"
    print(latex_table)




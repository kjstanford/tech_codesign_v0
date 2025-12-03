import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k, e, epsilon_0, hbar, m_e
from numpy import exp, log, sqrt, pi, abs, log10, tanh
from scipy.optimize import fsolve
import sympy
from src.sim_util import custom_sech, custom_coth

kB = k # Boltzmann's constant [JK^-1]
q = e # elementary charge [C]
m0 = m_e # electron mass [kg]
eps0 = epsilon_0 # vacuum permittivity [Fm^-1]
T = 300 # Room Temperature [K]
phit = kB*T/q # Thermal voltage at room temperature [V]
kT = kB*T # Thermal energy at room temperature [J]

def get_Lscale(eps_gox, eps_semi, tgox, tsemi):
    # NOTE: this equation comes from page 102 of "FinFET Modeling for IC Simulation and Design," but there is a typo
    # regarding the first term inside the square root. It should probably be changed to eps_semi / eps_gox instead of eps_gox / eps_semi.
    # So we have done that here.
    return sympy.sqrt( (eps_semi / eps_gox) * tgox * tsemi * ( 1 + eps_gox * tsemi / ( 4 * eps_semi * tgox ) ) )

def symbolic_Rsd_model_cmg(Lc, Lext, Wc, Wext, rho_c, Rsh_c, Rsh_ext):
    """
    Inputs:
    Lc: Contact length [m]
    Lext: Extension length [m]
    Wc: Contact width [m]
    Wext: Extension width [m]
    rho_c: Contact resistivity [Ohm*m^2]
    Rsh_c: Contact sheet resistance [Ohm/sq]
    Rsh_ext: Extension sheet resistance [Ohm/sq]
    """
    LT = sympy.sqrt(rho_c / Rsh_c)
    Rc = (rho_c / LT) * custom_coth(Lc / LT) / Wc
    Rext = Rsh_ext * Lext / Wext
    Rsd = Rc + Rext
    return Rsd

def symbolic_sce_model_cmg(Leff, Vt0, Lscale):
    """
    Inputs:
    Leff: Effective channel length [m]
    Vt0: Long channel threshold voltage [V]
    Lscale: SCE scale length [m]
    """
    n0 = 1 / (1 - custom_sech(Leff/(2*Lscale)))
    delta = 0.5 * custom_sech(Leff/(2*Lscale))
    dVt = Vt0 * custom_sech(Leff/(2*Lscale))
    return n0, delta, dVt

def symbolic_Cpar_model_cmg(Weff, Lext, eps_cap, tgate):
    """
    Inputs:
    Weff: Effective channel width [m]
    Lext: Extension length [m]
    eps_cap: Gate capacitance dielectric constant [F/m]
    tgate: Gate metal thickness [m]
    """
    Cpar = (eps_cap * eps0 / Lext) * Weff * tgate
    return Cpar

def symbolic_mvs_model(Vgs, Vds, Vt0, Leff, Weff, mD, mu_eff, vT, Cgc_on, n0, delta, dVt, Rs, Rd):
    """ Return symbolic expression for Id for codesign purposes """
    """
    Inputs:
    Vgs : Gate-to-source voltage [V]
    Vds : Drain-to-source voltage [V]
    Vt0  : Long channel threshold voltage [V]
    Leff : Effective channel length [m]
    Weff : Effective channel width [m]
    mD  : DoS effective mass [kg]
    mu_eff : Effective long channel mobility [m^2/Vs]
    vT  : Thermal velocity [m/s]
    Cgc_on : Gate-to-channel capacitance in the ON state (excluding DoS capacitance) [F/m^2]
    n0 : Subthreshold slope factor at Vds=0 [unit-less]
    delta : DIBL coefficient [unit-less]
    dVt : Threshold voltage rolloff [V]
    Rs : Source-side series resistance [Ohm]
    Rd : Drain-side series resistance [Ohm]
    """
    ksee = 0.5  # default value
    beta = 2.0  # default value
    theta = 1.0  # default value

    n = n0
    Vt = Vt0 - dVt - delta * Vds
    C2D = q**2 * mD / (pi * hbar**2)
    lambda_eff = 2 * phit * mu_eff / vT
    Lcrit_sat = ksee * Leff
    Tx = lambda_eff / (lambda_eff + Lcrit_sat)
    Cgc_on_eff = Cgc_on * C2D * (1 - Tx/2) / ( C2D * (1 - Tx/2) + n0 * Cgc_on )
    alpha = n0 * sympy.log( C2D * (1 - Tx/2) / (n0 * Cgc_on_eff) )
    vxo = vT * lambda_eff / ( lambda_eff + 2 * ksee * Leff )
    mu_eff_adj = 2 * mu_eff * ( ( lambda_eff + ksee*Leff ) / ( lambda_eff + 2*ksee*Leff ) ) * (Leff / ( Leff + lambda_eff ))

    Ff = 1 / ( 1 + sympy.exp( ( Vgs - ( Vt - 0.5 * alpha * phit ) ) / ( alpha * phit ) ) )
    Qix0 = Cgc_on_eff * n * phit * sympy.log( 1 + sympy.exp( (Vgs - ( Vt - alpha*phit*Ff ) ) / (n * phit) ) )
    v = ( Ff + (1 - Ff) / (1 + Weff * Rs * Cgc_on_eff * (1 - 2*delta) * vxo) ) * vxo
    Vdsats = v * ( Leff / mu_eff_adj + Weff * (Rs + Rd) * Qix0 )
    Vdsat = Vdsats * ( 1 - Ff ) + phit * Ff
    Fs = ( Vds / Vdsat ) / ( 1 + ( Vds / Vdsat )**beta )**(1/beta)
    Id = Weff * Qix0 * v * Fs

    return Id

def symbolic_delay_model(Vdd, Vt0, Lg, Wg, beta_p_n, mD_fac, mu_eff_n, mu_eff_p, eps_gox, tgox, eps_semi, tsemi, Lext, Lc, eps_cap, rho_c_n, rho_c_p, Rsh_c_n, Rsh_c_p, Rsh_ext_n, Rsh_ext_p, FO, M):
    """
    Inputs:
    Vdd : Supply voltage [V]
    Vt0 : Long channel threshold voltage [V]
    Lg : Gate length [m]
    Wg : Gate width [m]
    beta_p_n : Ratio of PMOS to NMOS widths [unit-less]
    mD_fac : DoS effective mass factor [unit-less]
    mu_eff_n : Effective long channel mobility for NMOS [m^2/Vs]
    mu_eff_p : Effective long channel mobility for PMOS [m^2/Vs]
    eps_gox : Gate oxide dielectric constant [F/m]
    tgox : Gate oxide thickness [m]
    eps_semi : Semiconductor dielectric constant [F/m]
    tsemi : Semiconductor thickness [m]
    Lext : Extension length [m]
    Lc : Contact length [m]
    eps_cap : Gate capacitance dielectric constant [F/m]
    rho_c_n : Contact resistivity for NMOS [Ohm*m^2]
    rho_c_p : Contact resistivity for PMOS [Ohm*m^2]
    Rsh_c_n : Contact sheet resistance for NMOS [Ohm/sq]
    Rsh_c_p : Contact sheet resistance for PMOS [Ohm/sq]
    Rsh_ext_n : Extension sheet resistance for NMOS [Ohm/sq]
    Rsh_ext_p : Extension sheet resistance for PMOS [Ohm/sq]
    FO : Fan-out [unit-less]
    M: Miller capacitance factor [unit-less]
    """
    Wc_n = Wg
    Wext_n = Wg # CORRECTION from 2*Wg to Wg
    Rs_n = symbolic_Rsd_model_cmg(Lc, Lext, Wc_n, Wext_n, rho_c_n, Rsh_c_n, Rsh_ext_n)
    Rd_n = symbolic_Rsd_model_cmg(Lc, Lext, Wc_n, Wext_n, rho_c_n, Rsh_c_n, Rsh_ext_n)

    Wc_p = beta_p_n * Wg
    Wext_p = beta_p_n * Wg # CORRECTION from 2*beta_p_n*Wg to beta_p_n*Wg
    Rs_p = symbolic_Rsd_model_cmg(Lc, Lext, Wc_p, Wext_p, rho_c_p, Rsh_c_p, Rsh_ext_p)
    Rd_p = symbolic_Rsd_model_cmg(Lc, Lext, Wc_p, Wext_p, rho_c_p, Rsh_c_p, Rsh_ext_p)

    Lscale = get_Lscale(eps_gox, eps_semi, tgox, tsemi)

    Leff = Lg
    Weff_Id_n = 2 * Wg
    Weff_Id_p = 2 * beta_p_n * Wg
    n0, delta, dVt = symbolic_sce_model_cmg(Leff, Vt0, Lscale)
    Cgc_on = eps_gox * eps0 / tgox
    mD = mD_fac * m0
    vT = sympy.sqrt(2 * kT * mD / (pi * mD**2))

    Ilow_n = symbolic_mvs_model(Vdd/2, Vdd, Vt0, Leff, Weff_Id_n, mD, mu_eff_n, vT, Cgc_on, n0, delta, dVt, Rs_n, Rd_n)
    Ihigh_n = symbolic_mvs_model(Vdd, Vdd/2, Vt0, Leff, Weff_Id_n, mD, mu_eff_n, vT, Cgc_on, n0, delta, dVt, Rs_n, Rd_n)
    Ieff_n = ( Ilow_n + Ihigh_n ) / 2

    Ilow_p = symbolic_mvs_model(Vdd/2, Vdd, Vt0, Leff, Weff_Id_p, mD, mu_eff_p, vT, Cgc_on, n0, delta, dVt, Rs_p, Rd_p)
    Ihigh_p = symbolic_mvs_model(Vdd, Vdd/2, Vt0, Leff, Weff_Id_p, mD, mu_eff_p, vT, Cgc_on, n0, delta, dVt, Rs_p, Rd_p)
    Ieff_p = ( Ilow_p + Ihigh_p ) / 2

    tgate = 2 * Lg
    Weff_Cpar_n = 2 * Wg
    Cpar_n = symbolic_Cpar_model_cmg(Weff_Cpar_n, Lext, eps_cap, tgate)
    Weff_Cpar_p = 2 * beta_p_n * Wg
    Cpar_p = symbolic_Cpar_model_cmg(Weff_Cpar_p, Lext, eps_cap, tgate)

    Cload_n = FO * ( (2/3) * Cgc_on * Weff_Id_n * Lg + Cpar_n ) + M * Cpar_n
    Cload_p = FO * ( (2/3) * Cgc_on * Weff_Id_p * Lg + Cpar_p ) + M * Cpar_p
    Cload = Cload_n + Cload_p

    tdelay = ( Cload / 2 ) * ( 1 / Ieff_n + 1 / Ieff_p ) * Vdd # CORRECTION: added Vdd term

    return tdelay, Ieff_n, Ieff_p, Cload

def symbolic_area_model(Lg, Wg, beta_p_n, Lext, Lc):
    """
    Inputs:
    Lg : Gate length [m]
    Wg : Gate width [m]
    beta_p_n : Ratio of PMOS to NMOS widths [unit-less]
    Lext : Extension length [m]
    Lc : Contact length [m]
    """
    Atotal = ( Lg + 2*Lext + Lc ) * ( Wg + beta_p_n * Wg )

    return Atotal

def symbolic_power_model(Vdd, Vt0, Lg, Wg, beta_p_n, mD_fac, mu_eff_n, mu_eff_p, eps_gox, tgox, eps_semi, tsemi, Lext, Lc, eps_cap, rho_c_n, rho_c_p, Rsh_c_n, Rsh_c_p, Rsh_ext_n, Rsh_ext_p, FO, M, a):
    """
    Inputs:
    Vdd : Supply voltage [V]
    Vt0 : Long channel threshold voltage [V]
    Lg : Gate length [m]
    Wg : Gate width [m]
    beta_p_n : Ratio of PMOS to NMOS widths [unit-less]
    mD_fac : DoS effective mass factor [unit-less]
    mu_eff_n : Effective long channel mobility for NMOS [m^2/Vs]
    mu_eff_p : Effective long channel mobility for PMOS [m^2/Vs]
    eps_gox : Gate oxide dielectric constant [F/m]
    tgox : Gate oxide thickness [m]
    eps_semi : Semiconductor dielectric constant [F/m]
    tsemi : Semiconductor thickness [m]
    Lext : Extension length [m]
    Lc : Contact length [m]
    eps_cap : Gate capacitance dielectric constant [F/m]
    rho_c_n : Contact resistivity for NMOS [Ohm*m^2]
    rho_c_p : Contact resistivity for PMOS [Ohm*m^2]
    Rsh_c_n : Contact sheet resistance for NMOS [Ohm/sq]
    Rsh_c_p : Contact sheet resistance for PMOS [Ohm/sq]
    Rsh_ext_n : Extension sheet resistance for NMOS [Ohm/sq]
    Rsh_ext_p : Extension sheet resistance for PMOS [Ohm/sq]

    FO : Fan-out [unit-less]
    M: Miller capacitance factor [unit-less]
    a : Activity factor [unit-less]
    """

    Wc_n = Wg
    Wext_n = Wg # CORRECTION from 2*Wg to Wg 
    Rs_n = symbolic_Rsd_model_cmg(Lc, Lext, Wc_n, Wext_n, rho_c_n, Rsh_c_n, Rsh_ext_n)
    Rd_n = symbolic_Rsd_model_cmg(Lc, Lext, Wc_n, Wext_n, rho_c_n, Rsh_c_n, Rsh_ext_n)

    Wc_p = beta_p_n * Wg
    Wext_p = beta_p_n * Wg # CORRECTION from 2*beta_p_n*Wg to beta_p_n*Wg
    Rs_p = symbolic_Rsd_model_cmg(Lc, Lext, Wc_p, Wext_p, rho_c_p, Rsh_c_p, Rsh_ext_p)
    Rd_p = symbolic_Rsd_model_cmg(Lc, Lext, Wc_p, Wext_p, rho_c_p, Rsh_c_p, Rsh_ext_p)

    Lscale = get_Lscale(eps_gox, eps_semi, tgox, tsemi)

    Leff = Lg
    Weff_Id_n = 2 * Wg
    Weff_Id_p = 2 * beta_p_n * Wg
    n0, delta, dVt = symbolic_sce_model_cmg(Leff, Vt0, Lscale)
    Cgc_on = eps_gox * eps0 / tgox
    mD = mD_fac * m0
    vT = sympy.sqrt(2 * kT * mD / (pi * mD**2))

    Ioff_n = symbolic_mvs_model(0, Vdd, Vt0, Leff, Weff_Id_n, mD, mu_eff_n, vT, Cgc_on, n0, delta, dVt, Rs_n, Rd_n)
    Ioff_p = symbolic_mvs_model(0, Vdd, Vt0, Leff, Weff_Id_p, mD, mu_eff_p, vT, Cgc_on, n0, delta, dVt, Rs_p, Rd_p)
    Ioff = Ioff_n + Ioff_p

    tgate = 2 * Lg
    Weff_Cpar_n = 2 * Wg
    Cpar_n = symbolic_Cpar_model_cmg(Weff_Cpar_n, Lext, eps_cap, tgate)
    Weff_Cpar_p = 2 * beta_p_n * Wg
    Cpar_p = symbolic_Cpar_model_cmg(Weff_Cpar_p, Lext, eps_cap, tgate)

    Cload_n = FO * ( (2/3) * Cgc_on * Weff_Id_n * Lg + Cpar_n ) + M * Cpar_n
    Cload_p = FO * ( (2/3) * Cgc_on * Weff_Id_p * Lg + Cpar_p ) + M * Cpar_p
    Cload = Cload_n + Cload_p

    Edynamic = a * Cload * Vdd**2
    Pstatic = Vdd * Ioff

    return Edynamic, Pstatic, Ioff_n, Ioff_p, Cload

def final_symbolic_models(Vdd, Vt0, Lg, Wg, beta_p_n, mD_fac, mu_eff_n, mu_eff_p, eps_gox, tgox, eps_semi, tsemi, Lext, Lc, eps_cap, rho_c_n, rho_c_p, Rsh_c_n, Rsh_c_p, Rsh_ext_n, Rsh_ext_p, FO, M, a):
    Area = symbolic_area_model(Lg, Wg, beta_p_n, Lext, Lc)
    Delay, Ieff_n, Ieff_p, Cload = symbolic_delay_model(Vdd, Vt0, Lg, Wg, beta_p_n, mD_fac, mu_eff_n, mu_eff_p, eps_gox, tgox, eps_semi, tsemi, Lext, Lc, eps_cap, rho_c_n, rho_c_p, Rsh_c_n, Rsh_c_p, Rsh_ext_n, Rsh_ext_p, FO, M)
    Edynamic, Pstatic, Ioff_n, Ioff_p, Cload = symbolic_power_model(Vdd, Vt0, Lg, Wg, beta_p_n, mD_fac, mu_eff_n, mu_eff_p, eps_gox, tgox, eps_semi, tsemi, Lext, Lc, eps_cap, rho_c_n, rho_c_p, Rsh_c_n, Rsh_c_p, Rsh_ext_n, Rsh_ext_p, FO, M, a)

    return Area, Delay, Edynamic, Pstatic, Ieff_n, Ieff_p, Ioff_n, Ioff_p, Cload

if __name__ == "__main__":
    Vdd, Vt0, Lg, Wg, beta_p_n, mD_fac, mu_eff_n, mu_eff_p, eps_gox, tgox, eps_semi, tsemi, Lext, Lc, eps_cap, rho_c_n, rho_c_p, Rsh_c_n, Rsh_c_p, Rsh_ext_n, Rsh_ext_p, FO, M, a = sympy.symbols('Vdd Vt0 Lg Wg beta_p_n mD_fac mu_eff_n mu_eff_p eps_gox tgox eps_semi tsemi Lext Lc eps_cap rho_c_n rho_c_p Rsh_c_n Rsh_c_p Rsh_ext_n Rsh_ext_p FO M a')
    final_Area, final_Delay, final_Edynamic, final_Pstatic, Ieff_n, Ieff_p, Ioff_n, Ioff_p, Cload = final_symbolic_models(Vdd, Vt0, Lg, Wg, beta_p_n, mD_fac, mu_eff_n, mu_eff_p, eps_gox, tgox, eps_semi, tsemi, Lext, Lc, eps_cap, rho_c_n, rho_c_p, Rsh_c_n, Rsh_c_p, Rsh_ext_n, Rsh_ext_p, FO, M, a)
    print("Final Symbolic Area Model:")
    # sympy.pprint(final_Area)
    print(final_Area)
    print("\nFinal Symbolic Delay Model:")
    # sympy.pprint(final_Delay)
    print(final_Delay)
    print("\nFinal Symbolic Power Model:")
    # sympy.pprint(final_Power)
    fclk_val = 1e9
    final_Power = final_Edynamic * fclk_val + final_Pstatic
    print(final_Power)

    # Example evaluation
    Vdd_val = 1
    Vt0_val = 0.5
    Lg_val = 40e-9
    Wg_val = 120e-9
    beta_p_n_val = 2
    mD_fac_val = 0.5
    mu_eff_n_val = 250e-4
    mu_eff_p_val = 125e-4
    eps_gox_val = 3.9
    tgox_val = 2.2e-9
    eps_semi_val = 11.7
    tsemi_val = 10e-9
    Lext_val = 10e-9
    Lc_val = 20e-9
    eps_cap_val = 3.9
    rho_c_n_val = 7e-12
    rho_c_p_val = 7e-12
    Rsh_c_n_val = 9000
    Rsh_c_p_val = 9000
    Rsh_ext_n_val = 9000
    Rsh_ext_p_val = 9000
    FO_val = 4
    M_val = 2
    a_val = 0.5

    final_Area_eval = final_Area.xreplace({Lg: Lg_val, Wg: Wg_val, beta_p_n: beta_p_n_val, Lext: Lext_val, Lc: Lc_val})

    final_Delay_eval = final_Delay.xreplace({Vdd: Vdd_val, Vt0: Vt0_val, Lg: Lg_val, Wg: Wg_val, beta_p_n: beta_p_n_val, mD_fac: mD_fac_val, mu_eff_n: mu_eff_n_val, mu_eff_p: mu_eff_p_val, eps_gox: eps_gox_val, tgox: tgox_val, eps_semi: eps_semi_val, tsemi: tsemi_val, Lext: Lext_val, Lc: Lc_val, eps_cap: eps_cap_val, rho_c_n: rho_c_n_val, rho_c_p: rho_c_p_val, Rsh_c_n: Rsh_c_n_val, Rsh_c_p: Rsh_c_p_val, Rsh_ext_n: Rsh_ext_n_val, Rsh_ext_p: Rsh_ext_p_val, FO: FO_val, M: M_val})

    final_Power_eval = final_Power.xreplace({Vdd: Vdd_val, Vt0: Vt0_val, Lg: Lg_val, Wg: Wg_val, beta_p_n: beta_p_n_val, mD_fac: mD_fac_val, mu_eff_n: mu_eff_n_val, mu_eff_p: mu_eff_p_val, eps_gox: eps_gox_val, tgox: tgox_val, eps_semi: eps_semi_val, tsemi: tsemi_val, Lext: Lext_val, Lc: Lc_val, eps_cap: eps_cap_val, rho_c_n: rho_c_n_val, rho_c_p: rho_c_p_val, Rsh_c_n: Rsh_c_n_val, Rsh_c_p: Rsh_c_p_val, Rsh_ext_n: Rsh_ext_n_val, Rsh_ext_p: Rsh_ext_p_val, FO: FO_val, M: M_val, a: a_val})

    print("\nEvaluated Area (m^2):")
    print(final_Area_eval)
    print("\nEvaluated Delay (s):")
    print(final_Delay_eval)
    print("\nEvaluated Power (W):")
    print(final_Power_eval)

    Ieff_n_eval = Ieff_n.xreplace({Vdd: Vdd_val, Vt0: Vt0_val, Lg: Lg_val, Wg: Wg_val, beta_p_n: beta_p_n_val, mD_fac: mD_fac_val, mu_eff_n: mu_eff_n_val, mu_eff_p: mu_eff_p_val, eps_gox: eps_gox_val, tgox: tgox_val, eps_semi: eps_semi_val, tsemi: tsemi_val, Lext: Lext_val, Lc: Lc_val, eps_cap: eps_cap_val, rho_c_n: rho_c_n_val, rho_c_p: rho_c_p_val, Rsh_c_n: Rsh_c_n_val, Rsh_c_p: Rsh_c_p_val, Rsh_ext_n: Rsh_ext_n_val, Rsh_ext_p: Rsh_ext_p_val, FO: FO_val, M: M_val})

    Ieff_p_eval = Ieff_p.xreplace({Vdd: Vdd_val, Vt0: Vt0_val, Lg: Lg_val, Wg: Wg_val, beta_p_n: beta_p_n_val, mD_fac: mD_fac_val, mu_eff_n: mu_eff_n_val, mu_eff_p: mu_eff_p_val, eps_gox: eps_gox_val, tgox: tgox_val, eps_semi: eps_semi_val, tsemi: tsemi_val, Lext: Lext_val, Lc: Lc_val, eps_cap: eps_cap_val, rho_c_n: rho_c_n_val, rho_c_p: rho_c_p_val, Rsh_c_n: Rsh_c_n_val, Rsh_c_p: Rsh_c_p_val, Rsh_ext_n: Rsh_ext_n_val, Rsh_ext_p: Rsh_ext_p_val, FO: FO_val, M: M_val})

    Ioff_n_eval = Ioff_n.xreplace({Vdd: Vdd_val, Vt0: Vt0_val, Lg: Lg_val, Wg: Wg_val, beta_p_n: beta_p_n_val, mD_fac: mD_fac_val, mu_eff_n: mu_eff_n_val, mu_eff_p: mu_eff_p_val, eps_gox: eps_gox_val, tgox: tgox_val, eps_semi: eps_semi_val, tsemi: tsemi_val, Lext: Lext_val, Lc: Lc_val, eps_cap: eps_cap_val, rho_c_n: rho_c_n_val, rho_c_p: rho_c_p_val, Rsh_c_n: Rsh_c_n_val, Rsh_c_p: Rsh_c_p_val, Rsh_ext_n: Rsh_ext_n_val, Rsh_ext_p: Rsh_ext_p_val, FO: FO_val, M: M_val})

    Ioff_p_eval = Ioff_p.xreplace({Vdd: Vdd_val, Vt0: Vt0_val, Lg: Lg_val, Wg: Wg_val, beta_p_n: beta_p_n_val, mD_fac: mD_fac_val, mu_eff_n: mu_eff_n_val, mu_eff_p: mu_eff_p_val, eps_gox: eps_gox_val, tgox: tgox_val, eps_semi: eps_semi_val, tsemi: tsemi_val, Lext: Lext_val, Lc: Lc_val, eps_cap: eps_cap_val, rho_c_n: rho_c_n_val, rho_c_p: rho_c_p_val, Rsh_c_n: Rsh_c_n_val, Rsh_c_p: Rsh_c_p_val, Rsh_ext_n: Rsh_ext_n_val, Rsh_ext_p: Rsh_ext_p_val, FO: FO_val, M: M_val})

    Cload_eval = Cload.xreplace({Vdd: Vdd_val, Vt0: Vt0_val, Lg: Lg_val, Wg: Wg_val, beta_p_n: beta_p_n_val, mD_fac: mD_fac_val, mu_eff_n: mu_eff_n_val, mu_eff_p: mu_eff_p_val, eps_gox: eps_gox_val, tgox: tgox_val, eps_semi: eps_semi_val, tsemi: tsemi_val, Lext: Lext_val, Lc: Lc_val, eps_cap: eps_cap_val, rho_c_n: rho_c_n_val, rho_c_p: rho_c_p_val, Rsh_c_n: Rsh_c_n_val, Rsh_c_p: Rsh_c_p_val, Rsh_ext_n: Rsh_ext_n_val, Rsh_ext_p: Rsh_ext_p_val, FO: FO_val, M: M_val})

    print("\nEvaluated Ieff_n (A):")
    print(Ieff_n_eval)
    print("\nEvaluated Ieff_p (A):")
    print(Ieff_p_eval)
    print("\nEvaluated Ioff_n (A):")
    print(Ioff_n_eval)
    print("\nEvaluated Ioff_p (A):")
    print(Ioff_p_eval)
    print("\nEvaluated Cload (F):")
    print(Cload_eval)

    # debugging below
    Lscale = get_Lscale(eps_gox, eps_semi, tgox, tsemi)
    Lscale_eval = Lscale.xreplace({eps_gox: eps_gox_val, eps_semi: eps_semi_val, tgox: tgox_val, tsemi: tsemi_val})
    print("\nDebugging SCE Model Outputs:")
    print("Lscale:", Lscale_eval)
    n0_eval, delta_eval, dVt_eval = symbolic_sce_model_cmg(Lg_val, Vt0_val, Lscale_eval)
    print("n0:", n0_eval)
    print("delta:", delta_eval)
    print("dVt:", dVt_eval)
    print("effective Vt:", Vt0_val - dVt_eval - delta_eval * Vdd_val)

    # debugging Rsd model
    Rsd_n_eval = symbolic_Rsd_model_cmg(Lc_val, Lext_val, Wg_val, Wg_val, rho_c_n_val, Rsh_c_n_val, Rsh_ext_n_val)
    Rsd_p_eval = symbolic_Rsd_model_cmg(Lc_val, Lext_val, beta_p_n_val*Wg_val, beta_p_n_val*Wg_val, rho_c_p_val, Rsh_c_p_val, Rsh_ext_p_val)
    print("\nDebugging Rsd Model Outputs:")
    print("Rsd_n (Ohm-um):", Rsd_n_eval*Wg_val*1e6)
    print("Rsd_p (Ohm-um):", Rsd_p_eval*beta_p_n_val*Wg_val*1e6)
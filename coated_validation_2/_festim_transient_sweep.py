"""FESTIM 2 driver -- run under festim2-env:

    /opt/anaconda3/envs/festim2-env/bin/python _festim_transient_sweep.py

For each K_r value in a logarithmic sweep between Anderl 1992 clean-W and
Anderl 1999 native-oxide W, run a transient 2-layer FESTIM simulation of run
03.19 (100 nm W on 0.65 mm carbon steel, T = 547.7 K, p_up = 765 Torr).  The
upstream BC is SurfaceReactionBC with K_r the swept variable and K_d set by
detailed balance K_d = K_r * K_s_W^2.  The downstream BC is C = 0.

For SurfaceReactionBC FESTIM 2 only supports HydrogenTransportProblem (single
layer); we model the steel substrate's series resistance with a Robin-style
ParticleFluxBC drain at the W back face:

    f(c_W) = (D_st * K_s_st / (e_sub * K_s_W)) * c_W

Output: festim_sweep_traces.json  containing, for every K_r value:
        t_s, J_atomH (downstream flux in atom-H/m^2/s)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import festim as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

K_B_EV = 8.617_333_262e-5
N_A    = 6.022_140_76e23

# ---- material params (mirror transport_properties.py) ---------------------
W_PARAMS = dict(
    D0_b=5.2e-8, E_D_b=0.21,
    Ks0_b=1.7464e22,        # particle/m^3/Pa^0.5
    E_S_b=0.2788,
)
ST_PARAMS = dict(
    D0_s=1.397e-6,  E_D_s=0.3841,
    Ks0_s_mol=8.28e-3, E_S_s=-0.0616,   # mol units; converted at use
)

# ---- run conditions for 03.19 ---------------------------------------------
RUN_NAME = '03.19/run_1_14h40'
T        = 547.7030518848298
P_UP_PA  = 765.3470238813197 * 133.322
E_COAT   = 1e-7
E_SUB    = 6.5e-4

# Anderl bracket -- clean → native_oxide
ANDERL_CLEAN_KR0  = 3.2e-15        # m^4/particle/s
ANDERL_CLEAN_EKR  = 1.16
ANDERL_OXIDE_KR0  = 1.0e-17
ANDERL_OXIDE_EKR  = 1.50

# Sweep parameters
N_STEPS = 9                        # 9 log-spaced points across the bracket
T_END_S = 6 * 3600.0               # simulate 6 h to reach steady state
DT_S    = 60.0                     # 60-s steps; ~360 timesteps total


def kr_arrhenius_at_T(kr0, ekr, Tval):
    return kr0 * np.exp(-ekr / (K_B_EV * Tval))


def build_arrhenius_pair(scale_clean: float):
    """Return (K_r_0, E_K_r) such that the value at T = 547.7 K is
    `scale_clean` * Anderl-clean(547.7 K).  We hold E_K_r between the
    Anderl-clean (1.16 eV) and Anderl-oxide (1.50 eV) values via linear
    interpolation in log(scale_clean).
    """
    # interpolate E_K_r linearly in log10(scale)
    # scale = 1     -> Anderl clean       (E = 1.16)
    # scale = 1/360 -> Anderl oxide value at 547.7 K (E = 1.50)
    kr_clean_547 = kr_arrhenius_at_T(ANDERL_CLEAN_KR0, ANDERL_CLEAN_EKR, T)
    kr_oxide_547 = kr_arrhenius_at_T(ANDERL_OXIDE_KR0, ANDERL_OXIDE_EKR, T)
    bracket_log = np.log10(kr_oxide_547 / kr_clean_547)   # ~ -3.5 decades
    f = np.log10(scale_clean) / bracket_log               # 0 (clean) -> 1 (oxide)
    f = float(np.clip(f, 0.0, 1.0))
    E_K_r = ANDERL_CLEAN_EKR + f * (ANDERL_OXIDE_EKR - ANDERL_CLEAN_EKR)
    # solve K_r_0 from K_r at 547.7 K:
    K_r_target = scale_clean * kr_clean_547
    K_r_0 = K_r_target / np.exp(-E_K_r / (K_B_EV * T))
    return K_r_0, E_K_r


def build_model(kr0_part, E_kr):
    """SurfaceReactionBC at upstream, Robin substrate-sink at downstream.
    All units particle-based (FESTIM convention).
    """
    Ks_st_T = ST_PARAMS['Ks0_s_mol'] * np.exp(
        -ST_PARAMS['E_S_s'] / (K_B_EV * T)) * N_A          # particle/m^3/Pa^0.5
    Ks_W_T  = W_PARAMS ['Ks0_b']    * np.exp(
        -W_PARAMS ['E_S_b'] / (K_B_EV * T))                # particle/m^3/Pa^0.5
    D_st_T  = ST_PARAMS['D0_s']     * np.exp(
        -ST_PARAMS['E_D_s'] / (K_B_EV * T))
    drain_coeff = D_st_T * Ks_st_T / (E_SUB * Ks_W_T)

    mat_W = F.Material(D_0=W_PARAMS['D0_b'], E_D=W_PARAMS['E_D_b'],
                       K_S_0=W_PARAMS['Ks0_b'], E_K_S=W_PARAMS['E_S_b'])

    model = F.HydrogenTransportProblem()
    model.mesh = F.Mesh1D(vertices=np.linspace(0, E_COAT, 200))
    sd = F.VolumeSubdomain1D(id=1, material=mat_W, borders=[0, E_COAT])
    L  = F.SurfaceSubdomain1D(id=2, x=0)
    R_ = F.SurfaceSubdomain1D(id=3, x=E_COAT)
    model.subdomains = [sd, L, R_]
    H = F.Species(name='H', mobile=True)
    model.species = [H]
    model.temperature = T

    # detailed balance K_d = K_r * K_s_W^2  (particle units)
    Kd_0   = kr0_part * W_PARAMS['Ks0_b'] ** 2
    Ekd    = E_kr + 2 * W_PARAMS['E_S_b']
    surface_rx = F.SurfaceReactionBC(
        reactant=[H, H], gas_pressure=P_UP_PA,
        k_r0=kr0_part, E_kr=E_kr,
        k_d0=Kd_0,     E_kd=Ekd,
        subdomain=L,
    )
    drain_bc = F.ParticleFluxBC(
        subdomain=R_,
        value=lambda c_H: drain_coeff * c_H,
        species=H,
        species_dependent_value={'c_H': H},
    )
    model.boundary_conditions = [surface_rx, drain_bc]
    model.settings = F.Settings(atol=1e10, rtol=1e-6,
                                transient=True,
                                final_time=T_END_S)
    model.settings.stepsize = F.Stepsize(initial_value=DT_S)

    flux_export = F.SurfaceFlux(surface=R_, field=H)
    model.exports = [flux_export]
    return model, flux_export


def run_one(kr0_part, E_kr):
    model, fexport = build_model(kr0_part, E_kr)
    model.initialise()
    model.run()
    # SurfaceFlux time-series; fexport.t is times, fexport.data is values
    t = np.array(fexport.t)
    J = np.array(fexport.data)
    # downstream face has +n outward; flux of H leaving = +D*dc/dx > 0;
    # FESTIM's SurfaceFlux convention returns the negative of that, so take abs.
    return t, np.abs(J)


def main():
    scales = np.logspace(0, np.log10(1e-3), N_STEPS)   # 1 -> 1e-3 (clean->oxide-ish)
    out = []
    for scale in scales:
        K_r_0, E_K_r = build_arrhenius_pair(scale)
        K_r_at_T = kr_arrhenius_at_T(K_r_0, E_K_r, T)
        print(f'\n[FESTIM] scale={scale:.3e}  K_r_0={K_r_0:.3e} m^4/p/s  '
              f'E_K_r={E_K_r:.3f} eV  K_r({T:.1f}K)={K_r_at_T:.3e}')
        try:
            t, J = run_one(K_r_0, E_K_r)
            print(f'  J_steady = {J[-1]:.3e} atom/m^2/s   '
                  f'mol/m^2/s = {J[-1]/N_A:.3e}')
        except Exception as exc:
            print(f'  FAILED: {exc}')
            t, J = np.array([]), np.array([])
        out.append({
            'scale_vs_clean': float(scale),
            'K_r_0_particle': float(K_r_0),
            'E_K_r':          float(E_K_r),
            'K_r_at_run_T':   float(K_r_at_T),
            't_s':            t.tolist(),
            'J_atomH':        J.tolist(),
        })

    out_path = os.path.join(HERE, 'festim_sweep_traces.json')
    with open(out_path, 'w') as f:
        json.dump({
            'run_name': RUN_NAME,
            'T_K':      T,
            'p_up_Pa':  P_UP_PA,
            'e_coat':   E_COAT,
            'e_sub':    E_SUB,
            'sweep':    out,
        }, f, indent=2)
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()

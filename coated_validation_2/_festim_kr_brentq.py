"""FESTIM-side bisection: run with festim2-env

    /opt/anaconda3/envs/festim2-env/bin/python _festim_kr_brentq.py

Reads target_flux.json (written by step1_kr_fit.py BEFORE the bisection runs:
contains J_target_mol = the measured steady-state slope dP/dt × V / (R T A)
× 2 of run 03.19 expressed in atomic-H mol/m^2/s) and uses scipy.optimize
to find the K_r scale (vs Anderl 1992 clean) that best reproduces it via a
FESTIM SurfaceReaction simulation.

For each candidate scale we hold E_K_r linearly interpolated between
Anderl-clean (1.16 eV) and Anderl-1999 oxide (1.50 eV) -- exactly the same
1-parameter family used in _festim_transient_sweep.py.

Each FESTIM call is a 1-hour transient (steady state achieved long before
that for any K_r in the bracket).  Writes the best result to fitted_kr.json
and the full transient at the best K_r to festim_best_transient.json so the
analysis side can plot it.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import festim as F
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

K_B_EV = 8.617_333_262e-5
N_A    = 6.022_140_76e23

W_PARAMS = dict(D0_b=5.2e-8, E_D_b=0.21,
                Ks0_b=1.7464e22, E_S_b=0.2788)
ST_PARAMS = dict(D0_s=1.397e-6, E_D_s=0.3841,
                 Ks0_s_mol=8.28e-3, E_S_s=-0.0616)

T = 547.7030518848298
P_UP_PA = 765.3470238813197 * 133.322
E_COAT = 1e-7
E_SUB = 6.5e-4
T_FINAL_S = 3 * 3600.0
DT_S = 60.0

ANDERL_CLEAN_KR0  = 3.2e-15
ANDERL_CLEAN_EKR  = 1.16
ANDERL_OXIDE_KR0  = 1.0e-17
ANDERL_OXIDE_EKR  = 1.50


def kr_arrhenius_at_T(kr0, ekr, Tval):
    return kr0 * np.exp(-ekr / (K_B_EV * Tval))


def build_arrhenius_pair(scale_clean: float):
    kr_clean_T = kr_arrhenius_at_T(ANDERL_CLEAN_KR0, ANDERL_CLEAN_EKR, T)
    kr_oxide_T = kr_arrhenius_at_T(ANDERL_OXIDE_KR0, ANDERL_OXIDE_EKR, T)
    bracket_log = np.log10(kr_oxide_T / kr_clean_T)
    f = np.log10(scale_clean) / bracket_log
    f = float(np.clip(f, 0.0, 1.0))
    E_K_r = ANDERL_CLEAN_EKR + f * (ANDERL_OXIDE_EKR - ANDERL_CLEAN_EKR)
    K_r_target = scale_clean * kr_clean_T
    K_r_0 = K_r_target / np.exp(-E_K_r / (K_B_EV * T))
    return float(K_r_0), float(E_K_r)


def build_model(kr0_part, E_kr):
    Ks_st_T = ST_PARAMS['Ks0_s_mol'] * np.exp(
        -ST_PARAMS['E_S_s'] / (K_B_EV * T)) * N_A
    Ks_W_T  = W_PARAMS ['Ks0_b']    * np.exp(
        -W_PARAMS ['E_S_b'] / (K_B_EV * T))
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

    Kd_0 = kr0_part * W_PARAMS['Ks0_b'] ** 2
    Ekd  = E_kr + 2 * W_PARAMS['E_S_b']
    model.boundary_conditions = [
        F.SurfaceReactionBC(reactant=[H, H], gas_pressure=P_UP_PA,
                            k_r0=kr0_part, E_kr=E_kr,
                            k_d0=Kd_0, E_kd=Ekd, subdomain=L),
        F.ParticleFluxBC(subdomain=R_, value=lambda c_H: drain_coeff * c_H,
                         species=H, species_dependent_value={'c_H': H}),
    ]
    model.settings = F.Settings(atol=1e10, rtol=1e-6,
                                transient=True, final_time=T_FINAL_S)
    model.settings.stepsize = F.Stepsize(initial_value=DT_S)

    flux_export = F.SurfaceFlux(surface=R_, field=H)
    model.exports = [flux_export]
    return model, flux_export


def run_festim_at(scale):
    K_r_0, E_K_r = build_arrhenius_pair(scale)
    model, fexport = build_model(K_r_0, E_K_r)
    model.initialise()
    model.run()
    t = np.array(fexport.t)
    J = np.abs(np.array(fexport.data))
    return t, J, K_r_0, E_K_r


def main():
    target = json.load(open(os.path.join(HERE, 'target_flux.json')))
    J_target_mol = float(target['J_target_mol'])
    J_target_part = J_target_mol * N_A
    print(f'Target steady-state flux: {J_target_mol:.3e} mol/m^2/s = '
          f'{J_target_part:.3e} atom-H/m^2/s')

    # objective: J_FESTIM(scale) - J_target  in atom-H units (FESTIM native)
    history = []
    def objective(log10_scale):
        scale = 10 ** log10_scale
        t, J, K_r_0, E_K_r = run_festim_at(scale)
        J_ss = float(J[-1])
        diff = J_ss - J_target_part
        history.append({'scale': scale, 'log10_scale': log10_scale,
                        'K_r_0': K_r_0, 'E_K_r': E_K_r,
                        'J_ss_atomH': J_ss, 'diff': diff})
        print(f'  scale={scale:.4e}  J_FESTIM={J_ss:.3e}  '
              f'diff={diff:+.3e} (target={J_target_part:.3e})')
        return diff

    # bracket: log10(scale) in [-3, 0]  (Anderl oxide-ish to Anderl clean)
    lo, hi = -3.5, 0.0
    f_lo = objective(lo)
    f_hi = objective(hi)
    if f_lo * f_hi > 0:
        # widen
        lo = -5.0
        f_lo = objective(lo)

    print(f'\nBrentq bracket: log10_scale in [{lo}, {hi}]   '
          f'f(lo)={f_lo:+.3e}  f(hi)={f_hi:+.3e}')
    log10_scale_best, res = brentq(objective, lo, hi, xtol=1e-3,
                                   maxiter=20, full_output=True)
    scale_best = 10 ** log10_scale_best
    print(f'\nBest log10(scale) = {log10_scale_best:.4f}  '
          f'(scale = {scale_best:.4e})  iterations = {res.iterations}')

    # one final FESTIM run at the optimum to record the full transient
    t_best, J_best, K_r_0_best, E_K_r_best = run_festim_at(scale_best)
    print(f'  at optimum:  K_r_0={K_r_0_best:.3e}  E_K_r={E_K_r_best:.3f}  '
          f'J_ss={J_best[-1]:.3e}')

    out_fitted = {
        'run_used':              '03.19/run_1_14h40',
        'T_K':                   T,
        'p_up_Pa':               P_UP_PA,
        'K_r_0_particle':        K_r_0_best,
        'K_r_0_mol':             K_r_0_best * N_A,
        'E_K_r':                 E_K_r_best,
        'scale_vs_anderl_clean': scale_best,
        'K_r_at_run_T_mol':      kr_arrhenius_at_T(
                                     K_r_0_best * N_A, E_K_r_best, T),
        'J_target_mol':          J_target_mol,
        'J_FESTIM_at_fit_mol':   float(J_best[-1] / N_A),
        'comment': ('K_r found by scipy.optimize.brentq on FESTIM SurfaceReaction '
                    'steady-state flux, target = SHIELD measured asymptotic '
                    'dP/dt for run 03.19.'),
    }
    with open(os.path.join(HERE, 'fitted_kr.json'), 'w') as f:
        json.dump(out_fitted, f, indent=2)
    with open(os.path.join(HERE, 'festim_best_transient.json'), 'w') as f:
        json.dump({'fit': out_fitted,
                   't_s': t_best.tolist(),
                   'J_atomH': J_best.tolist(),
                   'history': history}, f, indent=2)
    print(f'\nwrote fitted_kr.json and festim_best_transient.json')


if __name__ == '__main__':
    main()

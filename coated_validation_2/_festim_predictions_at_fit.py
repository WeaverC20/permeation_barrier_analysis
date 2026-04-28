"""FESTIM driver -- run with festim2-env:

    /opt/anaconda3/envs/festim2-env/bin/python _festim_predictions_at_fit.py

Uses the fitted K_r from fitted_kr.json to predict the steady-state flux
through the W coating + carbon-steel substrate at the (T, p) of each of the
two 274.6 C SHIELD runs.

Writes festim_at_fit.json with one record per run.
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

W_PARAMS = dict(D0_b=5.2e-8, E_D_b=0.21,
                Ks0_b=1.7464e22, E_S_b=0.2788)
ST_PARAMS = dict(D0_s=1.397e-6, E_D_s=0.3841,
                 Ks0_s_mol=8.28e-3, E_S_s=-0.0616)

RUNS = [
    ("03.19/run_1_14h40", 547.7030518848298, 765.3470238813197, 6.5e-4),
    ("03.23/run_1_17h00", 547.6689098586382, 388.5432404682812, 6.5e-4),
    ("03.26/run_1_12h03", 597.1643725826273, 399.94165293751087, 6.5e-4),
    ("03.27/run_1_08h51", 597.3513082931236, 395.9144582222676,  6.5e-4),
]
TORR_TO_PA = 133.322
E_COAT = 1e-7


def build_model(p_pa, e_sub, T, kr0_part, E_kr):
    Ks_st_T = ST_PARAMS['Ks0_s_mol'] * np.exp(
        -ST_PARAMS['E_S_s'] / (K_B_EV * T)) * N_A
    Ks_W_T  = W_PARAMS ['Ks0_b']    * np.exp(
        -W_PARAMS ['E_S_b'] / (K_B_EV * T))
    D_st_T  = ST_PARAMS['D0_s']     * np.exp(
        -ST_PARAMS['E_D_s'] / (K_B_EV * T))
    drain_coeff = D_st_T * Ks_st_T / (e_sub * Ks_W_T)

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
        F.SurfaceReactionBC(reactant=[H, H], gas_pressure=p_pa,
                            k_r0=kr0_part, E_kr=E_kr,
                            k_d0=Kd_0, E_kd=Ekd, subdomain=L),
        F.ParticleFluxBC(subdomain=R_, value=lambda c_H: drain_coeff * c_H,
                         species=H, species_dependent_value={'c_H': H}),
    ]
    model.settings = F.Settings(atol=1e-0, rtol=1e-10, transient=False)
    flux_R = F.SurfaceFlux(surface=R_, field=H)
    model.exports = [flux_R]
    return model, flux_R


def main():
    fit = json.load(open(os.path.join(HERE, 'fitted_kr.json')))
    K_r_0_part = fit['K_r_0_particle']
    E_K_r      = fit['E_K_r']
    print(f'Using fitted K_r:  K_r_0 = {K_r_0_part:.3e} m^4/particle/s, '
          f'E_K_r = {E_K_r:.3f} eV')

    out = []
    for name, T, p_torr, e_sub in RUNS:
        p_pa = p_torr * TORR_TO_PA
        print(f'\n[FESTIM] {name}  T={T:.1f}K  p={p_pa:.2e}Pa')
        model, flux_R = build_model(p_pa, e_sub, T, K_r_0_part, E_K_r)
        model.initialise()
        model.run()
        J = abs(float(flux_R.data[0]))
        print(f'  J_FESTIM = {J:.3e} atom/m^2/s = {J/N_A:.3e} mol/m^2/s')
        out.append({'name': name, 'T_K': T, 'p_Pa': p_pa,
                    'J_FESTIM_atomH': J, 'J_FESTIM_mol': J / N_A})

    out_path = os.path.join(HERE, 'festim_at_fit.json')
    with open(out_path, 'w') as f:
        json.dump({'fitted_kr': fit, 'runs': out}, f, indent=2)
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()

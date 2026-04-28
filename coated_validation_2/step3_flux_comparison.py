"""step3_flux_comparison.py
For the two 274.6 °C SHIELD W runs (03.19 and 03.23), compare the measured
permeation flux against the analytical 2-layer model and FESTIM, all using
the K_r fitted in step 1 plus the literature D, K_s pack defined in
transport_properties.py.

Six bars per run, all in mol H atoms / m^2 / s:
  - bare substrate           : Phi_st * sqrt(p) / e_sub
  - J_DL analytical          : 2-layer Sieverts/diffusion-limited closed form
                               (independent of K_d)
  - J_SL analytical          : K_d_W * p * R/(1+R)
  - J_full analytical quartic: J_DL * J*(W, R) from the quartic
  - J_SHIELD measured        : SHIELD CSV J value (TS-corrected)
  - J_FESTIM SurfaceReaction : steady-state FESTIM with the fitted K_r

Saves figs/step3_flux_comparison.pdf .
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import transport_properties as tp
from experimental_data import load_W_runs

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, 'figs')
os.makedirs(FIGDIR, exist_ok=True)

sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..',
                                                 'two_layer_analytical')))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'two_layer_analytical_mod',
    os.path.abspath(os.path.join(HERE, '..', 'two_layer_analytical',
                                 '2_layer_analytical.py')))
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
solve_two_layer = _m.solve_two_layer


def Kd_W_at_T(T, K_r_0_mol, E_K_r):
    Kr = K_r_0_mol * np.exp(-E_K_r / (tp.K_B_EV * T))
    Ks = tp.Ks('tungsten', T)
    return Kr * Ks * Ks


def Kd_steel_at_T(T):
    Kr = tp.Kr_steel(T)
    Ks = tp.Ks('carbon_steel', T)
    return Kr * Ks * Ks


def J_DL_2layer(run):
    """Diffusion-limited 2-layer flux (mol/m^2/s)."""
    D1, K1 = tp.D('tungsten', run.T),    tp.Ks('tungsten', run.T)
    D2, K2 = tp.D('carbon_steel', run.T), tp.Ks('carbon_steel', run.T)
    return D1 * D2 * np.sqrt(run.p_up_Pa) / (
        D2 * run.e_coat / K1 + D1 * run.e_sub / K2)


def J_SL_2layer(run, kd_W, R):
    return kd_W * run.p_up_Pa * R / (1.0 + R)


def J_full_quartic(run, kd_W, R):
    D1, K1 = tp.D('tungsten', run.T),    tp.Ks('tungsten', run.T)
    D2, K2 = tp.D('carbon_steel', run.T), tp.Ks('carbon_steel', run.T)
    Σ = run.e_coat / (D1 * K1) + run.e_sub / (D2 * K2)
    W = kd_W * np.sqrt(run.p_up_Pa) * Σ
    _, _, Jstar = solve_two_layer(W, R)
    if not np.isfinite(Jstar):
        return np.nan
    return J_DL_2layer(run) * Jstar


def J_substrate(run):
    return tp.Phi('carbon_steel', run.T) * np.sqrt(run.p_up_Pa) / run.e_sub


def _build_rows(runs, K_r_0, E_K_r, fes_by_name):
    rows = []
    for r in runs:
        kd_W   = Kd_W_at_T(r.T, K_r_0, E_K_r)
        kd_st  = Kd_steel_at_T(r.T)
        Rval   = kd_st / kd_W
        j_DL   = J_DL_2layer(r)
        j_SL   = J_SL_2layer(r, kd_W, Rval)
        j_full = J_full_quartic(r, kd_W, Rval)
        j_sub  = J_substrate(r)
        j_obs  = float(r.J_atomH.n) / tp.N_A
        j_obs_err = float(r.J_atomH.s) / tp.N_A
        j_fes  = fes_by_name.get(r.name, {}).get('J_FESTIM_mol', float('nan'))
        rows.append(dict(run=r,
                         kd_W=kd_W, kd_st=kd_st, R=Rval,
                         J_sub=j_sub, J_DL=j_DL, J_SL=j_SL,
                         J_full=j_full, J_obs=j_obs, J_obs_err=j_obs_err,
                         J_FESTIM=j_fes))
    return rows


def _plot_group(rows, fname):
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    x = np.arange(len(rows))
    w = 0.13

    sub_color    = '#a5a5a5'
    DL_color     = '#5b9bd5'
    SL_color     = '#7f57c5'
    full_color   = '#2c8c4e'
    obs_color    = '#ed7d31'
    festim_color = '#d62728'

    ax.bar(x - 2.5 * w, [r['J_sub']   for r in rows], width=w,
           color=sub_color, edgecolor='k',
           label='J bare substrate (no coating)')
    ax.bar(x - 1.5 * w, [r['J_DL']    for r in rows], width=w,
           color=DL_color, edgecolor='k',
           label='J$_{DL}$ analytical')
    ax.bar(x - 0.5 * w, [r['J_SL']    for r in rows], width=w,
           color=SL_color, edgecolor='k',
           label='J$_{SL}$ analytical (fitted K$_r$)')
    ax.bar(x + 0.5 * w, [r['J_full']  for r in rows], width=w,
           color=full_color, edgecolor='k',
           label='J$_{full}$ 2-layer quartic (fitted K$_r$)')
    ax.bar(x + 1.5 * w, [r['J_obs']   for r in rows], width=w,
           yerr=[r['J_obs_err'] for r in rows],
           color=obs_color, edgecolor='k', capsize=3,
           label='J$_{SHIELD}$ measured')
    ax.bar(x + 2.5 * w, [r['J_FESTIM'] for r in rows], width=w,
           color=festim_color, edgecolor='k', hatch='//',
           label='J$_{FESTIM}$ SurfaceReaction (fitted K$_r$)')

    ax.set_yscale('log')
    pos = np.array([row[k] for row in rows for k in
                    ('J_sub', 'J_DL', 'J_SL', 'J_full', 'J_obs', 'J_FESTIM')])
    pos = pos[np.isfinite(pos) & (pos > 0)]
    ax.set_ylim(pos.min() / 5, pos.max() * 8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f'{r["run"].name.split("/")[0]}\n'
         f'T = {r["run"].T-273.15:.0f}°C, '
         f'p = {r["run"].p_up_Pa/133.322:.0f} Torr'
         for r in rows], fontsize=11)
    ax.set_ylabel(r'Permeation flux $J$  [mol H atoms / m$^2$ / s]',
                  fontsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(True, which='both', axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    for i, r in enumerate(rows):
        prf = r['J_sub'] / r['J_obs']
        ax.text(x[i] - 2.5 * w, r['J_sub'] * 1.6,
                f'PRF={prf:.1f}', ha='center', fontsize=9,
                color='black', fontweight='bold')
    fig.tight_layout()
    out = os.path.join(FIGDIR, fname)
    fig.savefig(out, dpi=160, bbox_inches='tight')
    print(f'wrote {fname}')


def _print_table(rows, header):
    print(f'\n=== {header} ===')
    print(f'{"run":24s} {"T(K)":>6s} {"p(Pa)":>10s} '
          f'{"K_d_W":>11s} {"R":>9s}  '
          f'{"J_sub":>11s} {"J_DL":>11s} {"J_SL":>11s} '
          f'{"J_full":>11s} {"J_obs":>11s} {"J_FESTIM":>11s}')
    for r in rows:
        print(f'{r["run"].name:24s} {r["run"].T:6.1f} '
              f'{r["run"].p_up_Pa:10.2e} {r["kd_W"]:11.2e} {r["R"]:9.2e}  '
              f'{r["J_sub"]:11.2e} {r["J_DL"]:11.2e} {r["J_SL"]:11.2e} '
              f'{r["J_full"]:11.2e} {r["J_obs"]:11.2e} {r["J_FESTIM"]:11.2e}')


def main():
    fit = json.load(open(os.path.join(HERE, 'fitted_kr.json')))
    K_r_0 = fit['K_r_0_mol']
    E_K_r = fit['E_K_r']
    print(f'Fitted K_r:  K_r_0 = {K_r_0:.3e} m^4/(mol s), E_K_r = {E_K_r:.3f} eV')

    festim = json.load(open(os.path.join(HERE, 'festim_at_fit.json')))
    fes_by_name = {rec['name']: rec for rec in festim['runs']}

    all_runs = load_W_runs()
    runs_275 = [r for r in all_runs if r.T < 580.0]
    runs_324 = [r for r in all_runs if r.T >= 580.0]

    rows_275 = _build_rows(runs_275, K_r_0, E_K_r, fes_by_name)
    rows_324 = _build_rows(runs_324, K_r_0, E_K_r, fes_by_name)

    _plot_group(rows_275, 'step3_flux_comparison_275C.pdf')
    _plot_group(rows_324, 'step3_flux_comparison_324C.pdf')

    # back-compat: also save the 275C plot as the original filename
    _plot_group(rows_275, 'step3_flux_comparison.pdf')

    _print_table(rows_275, '274.6 °C runs')
    _print_table(rows_324, '324 °C runs')


if __name__ == '__main__':
    main()

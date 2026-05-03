"""step2_regime_overlay.py
Place every 100 nm W-on-carbon-steel SHIELD run on the (W, R) regime map of
the 2-layer analytical model, using the K_r fitted in step 1.

W = K_d_W * sqrt(p_up) * (e_coat / (D_W K_s_W) + e_sub / (D_st K_s_st))
R = K_d_steel(T) / K_d_W(T)   with K_d = K_r * K_s^2 in detailed balance

Saves figs/step2_regime_overlay.pdf.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib.pyplot as plt

import transport_properties as tp
from experimental_data import load_W_runs

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, 'figs')
os.makedirs(FIGDIR, exist_ok=True)

# Re-use the analytical regime-map computation from the existing solver
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..',
                                                 'two_layer_analytical')))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'two_layer_analytical_mod',
    os.path.abspath(os.path.join(HERE, '..', 'two_layer_analytical',
                                 '2_layer_analytical.py')))
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
compute_regime_map = _m.compute_regime_map


def main():
    # ---- fitted K_r ---------------------------------------------------------
    fit = json.load(open(os.path.join(HERE, 'fitted_kr.json')))
    K_r_0 = fit['K_r_0_mol']        # m^4/(mol s)
    E_K_r = fit['E_K_r']            # eV
    print(f'K_r_0 = {K_r_0:.3e} m^4/(mol s),  E_K_r = {E_K_r:.3f} eV')

    def Kd_W(T):
        Kr = K_r_0 * np.exp(-E_K_r / (tp.K_B_EV * T))
        Ks = tp.Ks('tungsten', T)
        return Kr * Ks * Ks                                            # mol/m^2/s/Pa

    def Kd_steel(T):
        Kr = tp.Kr_steel(T)
        Ks = tp.Ks('carbon_steel', T)
        return Kr * Ks * Ks

    def W_for_run(run):
        Σ = (run.e_coat / (tp.D('tungsten', run.T) * tp.Ks('tungsten', run.T))
             + run.e_sub  / (tp.D('carbon_steel', run.T) * tp.Ks('carbon_steel', run.T)))
        return Kd_W(run.T) * np.sqrt(run.p_up_Pa) * Σ

    def R_for_run(run):
        return Kd_steel(run.T) / Kd_W(run.T)

    # ---- regime-map background ---------------------------------------------
    W_AXIS = (1e-6, 1e6)
    R_AXIS = (1e-4, 1e4)
    W_range = np.logspace(*np.log10(W_AXIS), 110)
    R_range = np.logspace(*np.log10(R_AXIS), 60)
    Wg, Rg, errDL, errSL, _ = compute_regime_map(W_range, R_range)
    min_err = np.minimum(errDL, errSL)

    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    ax.contourf(Wg, Rg, np.where(errDL < errSL, 0.0, 1.0),
                levels=[-0.5, 0.5, 1.5],
                cmap=plt.colormaps.get_cmap('RdYlBu').resampled(2),
                alpha=0.55)
    ax.contourf(Wg, Rg, min_err, levels=[0.05, np.inf], colors=['white'],
                alpha=0.55)
    ax.contour(Wg, Rg, min_err, levels=[0.05], colors='black', linewidths=1.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(*W_AXIS); ax.set_ylim(*R_AXIS)
    ax.set_xlabel('$W = K_{d1}\\sqrt{p}\\,[e_1/(D_1 K_{s1}) + e_2/(D_2 K_{s2})]$',
                  fontsize=14)
    ax.set_ylabel('$R = K_{d2}/K_{d1}$', fontsize=14)
    ax.tick_params(labelsize=12)
    # SL / DL / mixed regime labels
    ax.text(W_AXIS[0] * 5, R_AXIS[0] * 3, 'SL', fontsize=22, color='navy',
            alpha=0.7, fontweight='bold', ha='left', va='bottom')
    ax.text(W_AXIS[1] / 5, 1.0, 'DL', fontsize=22, color='darkred',
            alpha=0.7, fontweight='bold', ha='right', va='center')
    ax.text(W_AXIS[1] / 5, R_AXIS[0] * 3, 'mixed', fontsize=14, color='gray',
            alpha=0.85, ha='right', va='bottom')
    ax.grid(True, which='both', alpha=0.2, linewidth=0.5)

    # ---- runs ---------------------------------------------------------------
    runs = load_W_runs()

    label_offsets = [(10, 10), (10, -22), (-110, 10), (-110, -22)]
    rows = []
    for j, r in enumerate(runs):
        Wv = W_for_run(r)
        Rv = R_for_run(r)
        rows.append((r, Wv, Rv))
        ax.scatter(Wv, Rv, marker='o', s=180, color='C0',
                   edgecolor='k', linewidth=1.5, zorder=5)
        T_C = r.T - 273.15
        ox, oy = label_offsets[j % len(label_offsets)]
        ax.annotate(f'T={T_C:.0f}°C, p={r.p_up_Pa/133.322:.0f} Torr',
                    (Wv, Rv), xytext=(ox, oy),
                    textcoords='offset points', fontsize=10,
                    arrowprops=dict(arrowstyle='-', color='black',
                                    lw=0.5, alpha=0.6))

    ax.legend(handles=[plt.Line2D([], [], marker='o', color='C0',
                                   markersize=11, markeredgecolor='k',
                                   linewidth=0,
                                   label='100 nm W on carbon steel')],
              loc='upper right', fontsize=12, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'step2_regime_overlay.pdf'),
                dpi=160, bbox_inches='tight')
    print(f'wrote step2_regime_overlay.pdf')

    # ---- console summary ---------------------------------------------------
    print(f'\n{"run":24s} {"T(K)":>6s} {"p(Pa)":>10s} '
          f'{"K_d_W":>11s} {"K_d_st":>11s} {"R":>9s} {"W":>10s}  regime')
    for r, Wv, Rv in rows:
        kdw = Kd_W(r.T)
        kds = Kd_steel(r.T)
        if Wv > 100:
            reg = 'DL'
        elif Wv < 0.01:
            reg = 'SL'
        else:
            reg = 'mixed'
        print(f'{r.name:24s} {r.T:6.1f} {r.p_up_Pa:10.2e} '
              f'{kdw:11.2e} {kds:11.2e} {Rv:9.2e} {Wv:10.2e}  {reg}')


if __name__ == '__main__':
    main()

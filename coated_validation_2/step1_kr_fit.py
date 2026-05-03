"""step1_kr_fit.py
Two-stage K_r fit for run 03.19:

  (a) compute the SHIELD measured asymptotic flux from the late-time slope
      dP/dt of P_down(t), and write it to target_flux.json so the FESTIM-
      side bisection driver can read it.

  (b) AFTER running

          /opt/anaconda3/envs/festim2-env/bin/python _festim_kr_brentq.py

      this script (re-run) reads festim_best_transient.json (FESTIM's best
      K_r and transient at the optimum) and produces

          figs/step1_kr_fit_transients.pdf  -- measured P_down(t) vs the
                                               single best-fit FESTIM curve

The flow is therefore:

    python step1_kr_fit.py                # writes target_flux.json
    /opt/anaconda3/envs/festim2-env/bin/python _festim_kr_brentq.py
    python step1_kr_fit.py                # reads festim_best_transient.json
                                          # and produces the PDF
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib.pyplot as plt

import transport_properties as tp
from experimental_data import (
    load_run_pressure_trace, AREA_M2, V_DOWNSTREAM_M3, R_GAS,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, 'figs')
os.makedirs(FIGDIR, exist_ok=True)

TARGET = os.path.join(HERE, 'target_flux.json')
BEST   = os.path.join(HERE, 'festim_best_transient.json')
SWEEP  = os.path.join(HERE, 'festim_sweep_traces.json')


def _load_and_baseline_trace():
    trace = load_run_pressure_trace('03.19/run_1_14h40')
    t  = trace['t_s']
    P  = trace['P_down_Pa']
    pre = (t < 0) & (t > -45)
    P_base = float(np.median(P[pre])) if pre.sum() > 5 else 0.0
    return trace, t, P - P_base, P_base


def _measured_steady_slope(t, P, t_window_h=(2.0, 6.0)):
    """Linear fit of P(t) over the late-time window to extract dP/dt_ss."""
    lo, hi = t_window_h[0] * 3600, t_window_h[1] * 3600
    sel = (t > lo) & (t < hi)
    if sel.sum() < 10:
        sel = t > 0
    slope, intercept = np.polyfit(t[sel], P[sel], 1)
    return float(slope), float(intercept), sel


def _slope_to_flux_atomicH_mol(dPdt_Pa_s, T_K):
    """dP/dt (Pa/s) → atomic-H flux (mol/m^2/s) via
        J_atomH = 2 * dP/dt * V / (R T A)
    (Baratron measures H_2; each H_2 dissociation produced 2 H atoms;
    those 2 atoms passing through the membrane give 1 H_2 molecule of dP).
    """
    return 2.0 * dPdt_Pa_s * V_DOWNSTREAM_M3 / (R_GAS * T_K * AREA_M2)


def _integrate_to_pressure(t_s, J_atomH_part, T_K):
    """Cumulative integral of J(t) (atom-H/m^2/s, particle units) -> P_down(t) Pa."""
    t = np.asarray(t_s)
    J_mol_atomH = np.asarray(J_atomH_part) / tp.N_A     # mol(H)/m^2/s
    J_mol_H2    = 0.5 * J_mol_atomH                      # mol(H_2)/m^2/s
    dPdt        = J_mol_H2 * AREA_M2 * R_GAS * T_K / V_DOWNSTREAM_M3
    P = np.cumsum(np.diff(np.r_[0, t]) * dPdt)
    return t, P


def main():
    trace, t_obs, P_obs0, P_base = _load_and_baseline_trace()
    T_K = trace['T_K']
    slope_meas, intercept_meas, sel = _measured_steady_slope(t_obs, P_obs0)
    J_target_mol = _slope_to_flux_atomicH_mol(slope_meas, T_K)
    print(f'  P_baseline           : {P_base:.3e} Pa')
    print(f'  measured asymptotic'
          f' dP/dt = {slope_meas:.3e} Pa/s  (window 2-6 h)')
    print(f'  -> J_target          : {J_target_mol:.3e} mol/m^2/s '
          f'= {J_target_mol*tp.N_A:.3e} atom-H/m^2/s')

    # write target file for the FESTIM bisection driver
    with open(TARGET, 'w') as f:
        json.dump({
            'run':            '03.19/run_1_14h40',
            'T_K':            T_K,
            'slope_meas_Pa_s': slope_meas,
            'intercept_meas': intercept_meas,
            'J_target_mol':   J_target_mol,
            'J_target_atomH': J_target_mol * tp.N_A,
            'comment': ('Linear fit of measured P_down(t) over 2-6 h '
                        'gives the asymptotic dP/dt; convert via '
                        '2 * V / (R T A) to atomic-H flux.'),
        }, f, indent=2)
    print(f'  wrote {os.path.basename(TARGET)}')

    # ---- if FESTIM bisection has run, plot the result ----------------------
    if not os.path.exists(BEST):
        print('\n  Now run the FESTIM bisection:')
        print('     /opt/anaconda3/envs/festim2-env/bin/python '
              '_festim_kr_brentq.py')
        print('  then re-run this script to produce the PDF.')
        return

    payload = json.load(open(BEST))
    fit  = payload['fit']
    t_f  = np.array(payload['t_s'])
    J_f  = np.array(payload['J_atomH'])
    t_p, P_pred = _integrate_to_pressure(t_f, J_f, T_K)

    print(f'\n  FESTIM optimum: K_r_0 = {fit["K_r_0_mol"]:.3e} m^4/(mol s),  '
          f'E_K_r = {fit["E_K_r"]:.3f} eV  '
          f'(scale = {fit["scale_vs_anderl_clean"]:.3e})')
    print(f'  J_target  = {fit["J_target_mol"]:.3e} mol/m^2/s')
    print(f'  J_FESTIM  = {fit["J_FESTIM_at_fit_mol"]:.3e} mol/m^2/s '
          f'(rel. err {(fit["J_FESTIM_at_fit_mol"]/fit["J_target_mol"]-1)*100:+.1f} %)')

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # plot the sweep of worse fits underneath, in light grey
    other_color = '#bdbdbd'
    if os.path.exists(SWEEP):
        sweep_payload = json.load(open(SWEEP))
        best_E = fit['E_K_r']
        # drop the best-fit entry, then keep every other one (~half as many)
        others = [e for e in sweep_payload.get('sweep', [])
                  if abs(e['E_K_r'] - best_E) >= 1e-6]
        others = others[::2]
        # cut the sweep traces a bit shorter than the best fit
        t_cut_s = float(t_p[-1]) * 0.92
        for entry in others:
            t_e = np.array(entry['t_s'])
            J_e = np.array(entry['J_atomH'])
            mask = t_e <= t_cut_s
            t_e, J_e = t_e[mask], J_e[mask]
            _, P_e = _integrate_to_pressure(t_e, J_e, T_K)
            ax.plot(t_e / 3600, P_e, color=other_color, lw=0.9, zorder=1.5)

    ax.plot(t_obs / 3600, P_obs0, color='k', lw=1.7,
            label='SHIELD measured (baseline-subtracted)', zorder=3)
    # show the linear-fit asymptote
    t_line = np.linspace(0, max(t_obs[-1], t_p[-1]) / 3600, 50)
    ax.plot(t_line, slope_meas * t_line * 3600 + intercept_meas,
            ls=':', color='gray', lw=1.4,
            label=f'linear fit of measured trace, slope = '
                  f'{slope_meas*3600:.3f} Pa/h', zorder=2.5)
    ax.plot(t_p / 3600, P_pred, color='red', lw=2.0,
            label=f'best-fit FESTIM SurfaceReaction\n'
                  f'$K_{{r,0}}$ = {fit["K_r_0_mol"]:.2e} m$^4$/(mol·s),  '
                  f'$E_{{K_r}}$ = {fit["E_K_r"]:.2f} eV', zorder=4)
    ax.set_xlabel('time since v3-open (hours)', fontsize=12)
    ax.set_ylabel('downstream P (Pa, baseline-subtracted)', fontsize=12)
    ax.tick_params(labelsize=11)

    # tighten the axes to the best-fit range so the red curve fills the frame
    t_end_h = float(t_p[-1] / 3600)
    P_max   = float(np.max(P_pred))
    ax.set_xlim(-0.03 * t_end_h, 1.02 * t_end_h)
    ax.set_ylim(-0.05 * P_max,    1.15 * P_max)

    # text annotations identifying the curves on the plot itself
    ax.text(t_end_h * 0.98, P_pred[-1] * 1.02, 'best fit',
            color='red', fontsize=11, fontweight='bold',
            ha='right', va='bottom')
    ax.text(t_end_h * 0.05, P_max * 1.10, 'other fits ($E_{K_r}$ sweep)',
            color='#7a7a7a', fontsize=11, fontstyle='italic',
            ha='left', va='center')

    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'step1_kr_fit_transients.pdf'),
                dpi=160, bbox_inches='tight')
    print(f'  wrote step1_kr_fit_transients.pdf')


if __name__ == '__main__':
    main()

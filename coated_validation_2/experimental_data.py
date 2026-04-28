"""coated_validation_2/experimental_data.py

Load:
  (i)  the four 100 nm W on carbon-steel SHIELD run rows (T, p_up, e_sub, J_obs)
       from ShieldRunsAnalysis/results/100nm_W_coated_carbon_steel_diffusivities.csv
       (with the 0.65-mm thickness correction baked in)

  (ii) the raw downstream-pressure time trace P_down(t) for run 03.19, used by
       step1_kr_fit.py to do a transient K_r fit against FESTIM.

Notation in this folder:
   t = 0  is the v3-open instant (recorded in run_metadata.json), i.e. the
   moment the upstream H_2 supply is connected to the hot sample.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from uncertainties import ufloat

# Re-use the existing voltage-to-Torr calibration helpers.
SHIELD_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'ShieldRunsAnalysis'))
if SHIELD_DIR not in sys.path:
    sys.path.insert(0, SHIELD_DIR)
from analysis_functions import (   # noqa: E402
    voltage_to_torr_baratron_downstream,
    voltage_to_torr_baratron_upstream,
    voltage_to_torr_wasp_downstream,
    voltage_to_temp_typeK,
)

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
SHIELD_RESULTS = os.path.join(REPO_ROOT, 'ShieldRunsAnalysis', 'results')
SHIELD_DATA    = '/Users/colinweaver/Documents/PTTEP/Results'

TORR_TO_PA = 133.322
NOTEBOOK_E_USED = 0.00136   # e hard-coded in SHIELD_analysis.ipynb

# Thicknesses recovered from run_metadata.json:
RUN_THICKNESS = {
    '03.19/run_1_14h40': 0.00065,
    '03.23/run_1_17h00': 0.00065,
    '03.26/run_1_12h03': 0.00065,
    '03.27/run_1_08h51': 0.00065,
}

# Downstream chamber volume (from SHIELD_analysis.ipynb).
V_DOWNSTREAM_M3 = 7.9e-5   # m^3
V_DOWNSTREAM_S  = 9.8e-6   # 1-sigma m^3
SAMPLE_DIAM_M   = 0.0155   # m
AREA_M2         = 0.25 * np.pi * SAMPLE_DIAM_M ** 2
R_GAS = 8.314              # J / mol / K


# --- CSV row loader ---------------------------------------------------------

_UFLOAT_RE = re.compile(r'\(([\d.+\-]+)\+/-([\d.+\-eE]+)\)e([\d+\-]+)')


def _parse_ufloat(s):
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    m = _UFLOAT_RE.match(s)
    if m:
        v, e, exp = m.groups()
        scale = 10.0 ** int(exp)
        return ufloat(float(v) * scale, float(e) * scale)
    try:
        return float(s)
    except ValueError:
        return np.nan


@dataclass
class WRun:
    name: str
    T: float          # K
    p_up_Pa: float
    e_sub: float      # m
    e_coat: float     # m   (always 1e-7 in this folder)
    J_atomH: object   # ufloat, atom-H/m^2/s, raw from CSV
    Phi_atomH: object # ufloat, atom-H/m/s/Pa^0.5, thickness-corrected
    D_eff: float      # m^2/s, single-slab timelag


def load_W_runs() -> list[WRun]:
    df = pd.read_csv(os.path.join(
        SHIELD_RESULTS, '100nm_W_coated_carbon_steel_diffusivities.csv'))
    out: list[WRun] = []
    for _, row in df.iterrows():
        name = row['Run Name']
        T = float(row['Temperature (K)'])
        p_torr = _parse_ufloat(row['Upstream Pressure (Torr)'])
        Phi    = _parse_ufloat(row['Permeability'])
        J      = _parse_ufloat(row['Flux'])
        D_eff  = float(row['Diffusivity'])
        e_real = RUN_THICKNESS.get(name, NOTEBOOK_E_USED)
        # Phi recorded is J*(NOTEBOOK_E_USED)/sqrt(p); rescale to e_real.
        if abs(e_real - NOTEBOOK_E_USED) > 1e-9:
            Phi = Phi * (e_real / NOTEBOOK_E_USED)
        out.append(WRun(
            name=name, T=T,
            p_up_Pa=float(getattr(p_torr, 'n', p_torr)) * TORR_TO_PA,
            e_sub=e_real, e_coat=1e-7,
            J_atomH=J, Phi_atomH=Phi, D_eff=D_eff,
        ))
    return out


def load_W_runs_275C() -> list[WRun]:
    return [r for r in load_W_runs() if r.T < 580.0]


# --- transient pressure trace for run 03.19 --------------------------------

def load_run_pressure_trace(run_name: str = '03.19/run_1_14h40'):
    """Load the downstream Baratron 1T pressure trace for the named run.

    Returns
    -------
    dict with keys:
      t_s         : time (s) zeroed at v3-open
      P_down_Pa   : downstream Baratron 1T pressure (Pa)
      P_up_Pa     : upstream Baratron 1KT pressure (Pa)
      T_K         : average sample temperature (K) over the latter half of
                    the trace
      e_sub       : substrate thickness (m) from metadata
      v3_open_s   : v3-open offset (s) relative to the CSV's first timestamp
    """
    base = os.path.join(SHIELD_DATA, run_name)
    meta = json.load(open(os.path.join(base, 'run_metadata.json')))
    df = pd.read_csv(os.path.join(base, 'shield_data.csv'))

    df['RealTimestamp'] = pd.to_datetime(df['RealTimestamp'], errors='coerce')
    t0 = df['RealTimestamp'].iloc[0]
    t_csv = (df['RealTimestamp'] - t0).dt.total_seconds().to_numpy()

    df = df.rename(columns=lambda x:
                   x.strip().replace(' ', '_').replace('(', '')
                            .replace(')', '').replace('/', '_per_'))
    V_baratron_down = df['Baratron626D_1T_Voltage_V'].to_numpy(float)
    V_baratron_up   = df['Baratron626D_1KT_Voltage_V'].to_numpy(float)
    V_TC            = df['furnace_thermocouple_Voltage_mV'].to_numpy(float)

    P_down_Pa = voltage_to_torr_baratron_downstream(V_baratron_down) * TORR_TO_PA
    P_up_Pa   = voltage_to_torr_baratron_upstream  (V_baratron_up  ) * TORR_TO_PA
    T_C       = voltage_to_temp_typeK(V_TC)
    T_K_mean  = float(np.mean(T_C[len(T_C)//2:])) + 273.15

    v3_open = datetime.fromisoformat(meta['run_info']['v3_open_time'])
    v3_open_s = (v3_open - t0.to_pydatetime()).total_seconds()

    e_sub = float(meta['run_info']['sample_thickness'])
    return {
        't_s': t_csv - v3_open_s,
        'P_down_Pa': P_down_Pa,
        'P_up_Pa':   P_up_Pa,
        'T_K':       T_K_mean,
        'e_sub':     e_sub,
        'v3_open_s': v3_open_s,
    }


def trace_to_flux(trace: dict) -> tuple[np.ndarray, np.ndarray]:
    """Convert downstream P_down(t) to instantaneous flux J(t) via

        J(t)  =  V / (R T A)  *  dP/dt(t)            [mol H atoms / m^2 / s]

    Hydrogen is detected as H_2 by the Baratron, so each H_2 molecule equals
    two H atoms; the conversion below already includes that factor (× 2).
    """
    t = trace['t_s']
    P = trace['P_down_Pa']
    T = trace['T_K']
    # smooth the pressure trace before differentiating
    if len(t) > 30:
        from scipy.signal import savgol_filter
        P_smooth = savgol_filter(P, window_length=31, polyorder=2)
    else:
        P_smooth = P
    dPdt = np.gradient(P_smooth, t)
    # mol(H2)/s = V/(RT) * dP/dt; mol(H)/s = 2 * mol(H2)/s
    J_mol = 2.0 * V_DOWNSTREAM_M3 / (R_GAS * T) * dPdt / AREA_M2
    return t, J_mol


if __name__ == '__main__':
    for r in load_W_runs():
        print(f'  {r.name}  T={r.T:.1f}K  p={r.p_up_Pa:.3e}Pa  '
              f'e_sub={r.e_sub*1e3:.2f}mm  Phi={r.Phi_atomH:.2e}')
    print()
    tr = load_run_pressure_trace()
    print(f'  trace: t in [{tr["t_s"][0]:.1f}, {tr["t_s"][-1]:.1f}] s   '
          f'P_down in [{tr["P_down_Pa"].min():.3e}, '
          f'{tr["P_down_Pa"].max():.3e}] Pa   T_K={tr["T_K"]:.2f}')

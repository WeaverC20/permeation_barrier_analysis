"""
SHIELD pressure-rise FESTIM simulation
======================================

Models a 100 nm tungsten coating on carbon steel under SHIELD-like conditions
and compares the predicted downstream pressure rise (and time-varying
permeability) to experiment.

- Substrate (carbon steel) D and S taken from
  ShieldRunsAnalysis/results/steel_diffusivities.csv via Arrhenius fits.
- Coating: 100 nm of tungsten, properties from h_transport_materials
  (Zhou diffusivity, Esteban solubility).
- For each requested temperature, the script picks the closest experimental
  W-coated run from 100nm_W_coated_carbon_steel_diffusivities.csv.
- Saves figures to ShieldRunsAnalysis/results/figs/festim_pressure_rise/.
"""

import csv
import json
import os
import re
import sys
from pathlib import Path

import festim as F
import h_transport_materials as htm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from analysis_functions import (  # noqa: E402
    average_pressure_after_increase,
    load_downstream_data,
    voltage_to_temp_typeK,
    voltage_to_torr_baratron_downstream,
    voltage_to_torr_baratron_upstream,
)


# =============================================================================
# USER INPUTS
# =============================================================================
TEMPERATURES_K = [550.0, 650.0]            # change this list to add more
BARRIER_THICKNESS = 1.0e-7                 # 100 nm tungsten
SUBSTRATE_THICKNESS = 6.5e-4               # 650 micron carbon steel
SAMPLE_DIAMETER = 0.0155                   # m
DOWNSTREAM_VOLUME = 7.9e-5                 # m^3 (downstream chamber)
RESULTS_PATH = Path("/Users/colinweaver/Documents/PTTEP/Results")

# =============================================================================
# CONSTANTS
# =============================================================================
K_B = 8.617333262e-5                       # eV/K
R_GAS = 8.314                              # J/(mol*K)
N_AVOGADRO = 6.022e23
TORR_PER_PA = 1.0 / 133.322
AREA = 0.25 * np.pi * SAMPLE_DIAMETER**2

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
FIGS_DIR = SCRIPT_DIR / "results" / "figs" / "festim_pressure_rise"
FIGS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Substrate properties (Arrhenius fits of carbon steel CSV)
# =============================================================================
def load_arrhenius_fits(csv_path):
    """Return (D_0, E_D, Phi_0, E_Phi) from steel_diffusivities.csv."""
    T_data, D_data, Phi_data = [], [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["Diffusivity"] or not row["Permeability"]:
                continue
            T_data.append(float(row["Temperature (K)"]))
            D_data.append(float(row["Diffusivity"]))

            perm_str = row["Permeability"].strip()
            m = re.match(
                r"\(?([0-9.\-eE]+)\+/-([0-9.\-eE]+)\)?(?:e([+\-]?[0-9]+))?",
                perm_str,
            )
            if m:
                val = float(m.group(1))
                exp = int(m.group(3)) if m.group(3) else 0
                Phi_data.append(val * 10**exp)
            else:
                Phi_data.append(float(perm_str))

    T = np.array(T_data)
    D = np.array(D_data)
    Phi = np.array(Phi_data)

    m_D, b_D = np.polyfit(1.0 / T, np.log(D), 1)
    D_0 = float(np.exp(b_D))
    E_D = float(-m_D * K_B)

    m_Phi, b_Phi = np.polyfit(1.0 / T, np.log(Phi), 1)
    Phi_0 = float(np.exp(b_Phi))
    E_Phi = float(-m_Phi * K_B)

    return D_0, E_D, Phi_0, E_Phi


SUB_CSV = SCRIPT_DIR / "results" / "steel_diffusivities.csv"
D_0_SUB, E_D_SUB, PHI_0_SUB, E_PHI_SUB = load_arrhenius_fits(SUB_CSV)
S_0_SUB = PHI_0_SUB / D_0_SUB
E_S_SUB = E_PHI_SUB - E_D_SUB

# =============================================================================
# Tungsten properties (h_transport_materials)
# =============================================================================
D_W = htm.diffusivities.filter(material="tungsten").filter(isotope="h").filter(
    author="zhou"
)[0]
S_W = htm.solubilities.filter(material="tungsten").filter(isotope="h").filter(
    author="esteban"
)[0]
D_0_W = float(D_W.pre_exp.magnitude)
E_D_W = float(D_W.act_energy.magnitude)
S_0_W = float(S_W.pre_exp.magnitude)
E_S_W = float(S_W.act_energy.magnitude)


# =============================================================================
# Experimental run lookup (W-coated CSV)
# =============================================================================
def load_w_coated_runs(csv_path):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("Temperature (K)"):
                continue
            rows.append(
                {
                    "name": row["Run Name"],
                    "T": float(row["Temperature (K)"]),
                    "P_up_Torr": float(row.get("Upstream Pressure (Torr)") or 0),
                }
            )
    return rows


W_RUNS = load_w_coated_runs(
    SCRIPT_DIR / "results" / "100nm_W_coated_carbon_steel_diffusivities.csv"
)


def closest_run(target_T):
    return min(W_RUNS, key=lambda r: abs(r["T"] - target_T))


# =============================================================================
# Experimental loader (handles both new shield_data.csv and old format)
# =============================================================================
def load_experiment(run_name):
    data_path = RESULTS_PATH / run_name
    metadata = json.loads((data_path / "run_metadata.json").read_text())

    if (data_path / "shield_data.csv").exists():
        t, df = load_downstream_data(str(data_path / "shield_data.csv"))
    else:
        t, df = load_downstream_data(str(data_path / "pressure_gauge_data.csv"))

    if "Baratron626D_1T_Voltage_V" in df.columns:
        v_b_down = df["Baratron626D_1T_Voltage_V"].to_numpy(dtype=float)
        v_b_up = df["Baratron626D_1KT_Voltage_V"].to_numpy(dtype=float)
    else:
        v_b_down = df["Baratron626D_1T_Voltage (V)"].to_numpy(dtype=float)
        v_b_up = df["Baratron626D_1KT_Voltage (V)"].to_numpy(dtype=float)

    P_b_down_Torr = voltage_to_torr_baratron_downstream(v_b_down)
    P_b_up_Torr = voltage_to_torr_baratron_upstream(v_b_up)

    # Temperature: prefer in-file thermocouple, fall back to thermocouple_data.csv
    if "furnace_thermocouple_Voltage_mV" in df.columns:
        T_mv = df["furnace_thermocouple_Voltage_mV"].to_numpy(dtype=float)
    elif "furnace_thermocouple_Voltage (mV)" in df.columns:
        T_mv = df["furnace_thermocouple_Voltage (mV)"].to_numpy(dtype=float)
    elif (data_path / "thermocouple_data.csv").exists():
        _, T_df = load_downstream_data(str(data_path / "thermocouple_data.csv"))
        if "type_K_thermocouple_Voltage_mV" in T_df.columns:
            T_mv = T_df["type_K_thermocouple_Voltage_mV"].to_numpy(dtype=float)
        else:
            T_mv = T_df["type K thermocouple_Voltage (mV)"].to_numpy(dtype=float)
    else:
        T_mv = None

    if T_mv is not None and len(T_mv) > 0:
        T_C = voltage_to_temp_typeK(T_mv)
        T_K = float(np.mean(T_C[len(T_C) // 4 :])) + 273.15
    else:
        T_K = metadata["run_info"]["furnace_setpoint"] + 273.15

    # Align t=0 with v3_open_time (upstream applied)
    start_str = metadata["run_info"]["start_time"]
    v3_open_str = metadata["run_info"].get("v3_open_time")
    if v3_open_str is not None:
        t_offset = (
            pd.to_datetime(v3_open_str) - pd.to_datetime(start_str)
        ).total_seconds()
    else:
        t_offset = 0.0

    t_shifted = t - t_offset
    P_up_Torr = float(average_pressure_after_increase(t, P_b_up_Torr))

    return {
        "t": t_shifted,
        "P_down_Torr": P_b_down_Torr,
        "P_up_Torr": P_up_Torr,
        "T_K": T_K,
        "metadata": metadata,
    }


# =============================================================================
# FESTIM: transient two-layer simulation (W barrier + steel)
# =============================================================================
def run_festim(T_K, P_up_Pa, final_time, max_stepsize=200.0):
    """Run a transient two-layer permeation simulation, return (t, J)."""
    barrier_pts = np.linspace(0, BARRIER_THICKNESS, 40)
    substrate_pts = np.linspace(
        BARRIER_THICKNESS, BARRIER_THICKNESS + SUBSTRATE_THICKNESS, 200
    )
    vertices = np.concatenate((barrier_pts, substrate_pts[1:]))

    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.mesh = F.Mesh1D(vertices=vertices)

    mat_b = F.Material(
        D_0=D_0_W,
        E_D=E_D_W,
        K_S_0=S_0_W,
        E_K_S=E_S_W,
        solubility_law="sieverts",
    )
    mat_s = F.Material(
        D_0=D_0_SUB,
        E_D=E_D_SUB,
        K_S_0=S_0_SUB,
        E_K_S=E_S_SUB,
        solubility_law="sieverts",
    )

    barrier = F.VolumeSubdomain1D(id=1, material=mat_b, borders=[0, BARRIER_THICKNESS])
    substrate = F.VolumeSubdomain1D(
        id=2,
        material=mat_s,
        borders=[BARRIER_THICKNESS, BARRIER_THICKNESS + SUBSTRATE_THICKNESS],
    )
    left = F.SurfaceSubdomain1D(id=3, x=0)
    right = F.SurfaceSubdomain1D(id=4, x=BARRIER_THICKNESS + SUBSTRATE_THICKNESS)
    interface = F.Interface(id=5, subdomains=[barrier, substrate], penalty_term=1e19)

    my_model.interfaces = [interface]
    my_model.subdomains = [barrier, substrate, left, right]

    H = F.Species(name="H", mobile=True, subdomains=[barrier, substrate])
    my_model.species = [H]
    my_model.surface_to_volume = {left: barrier, right: substrate}

    my_model.temperature = T_K

    sieverts_bc = F.SievertsBC(
        subdomain=left,
        S_0=mat_b.K_S_0,
        E_S=mat_b.E_K_S,
        pressure=P_up_Pa,
        species=H,
    )
    my_model.boundary_conditions = [
        sieverts_bc,
        F.FixedConcentrationBC(subdomain=right, value=0, species=H),
    ]

    flux_right = F.SurfaceFlux(surface=right, field=H)
    my_model.exports = [flux_right]

    my_model.settings = F.Settings(
        atol=1e10,
        rtol=1e-8,
        transient=True,
        final_time=final_time,
        max_iterations=200,
    )
    my_model.settings.stepsize = F.Stepsize(
        initial_value=1.0,
        growth_factor=1.05,
        cutback_factor=0.5,
        max_stepsize=max_stepsize,
        target_nb_iterations=20,
    )

    my_model.initialise()
    my_model.run()

    t_sim = np.asarray(flux_right.t)
    J_sim = np.abs(np.asarray(flux_right.data))
    return t_sim, J_sim


# =============================================================================
# Pressure rise and permeability conversions
# =============================================================================
def flux_to_pressure_rise_Torr(t_sim, J_sim, T_sample_K):
    """Integrate atom flux to a downstream pressure rise (Torr).

    Uses the same (single-particle) bookkeeping as the time-varying
    permeability script so the predicted P(t) can be compared directly with
    the Baratron trace.
    """
    dPdt_Pa = J_sim * R_GAS * T_sample_K * AREA / (DOWNSTREAM_VOLUME * N_AVOGADRO)
    P_Pa = cumulative_trapezoid(dPdt_Pa, t_sim, initial=0.0)
    return P_Pa * TORR_PER_PA


def permeability_from_dPdt(dPdt_Pa, P_up_Pa, T_sample_K, thickness):
    """Convert a downstream dP/dt to apparent permeability using the same
    convention as SHIELD_analysis_timevarying_perm.ipynb.
    """
    J_atoms = dPdt_Pa * DOWNSTREAM_VOLUME * N_AVOGADRO / (R_GAS * T_sample_K * AREA)
    return J_atoms * thickness / np.sqrt(P_up_Pa)


def smooth_dPdt(t, P_Torr, window=155, poly=3):
    if len(t) < window:
        return np.gradient(P_Torr, t) * 133.322
    if window % 2 == 0:
        window += 1
    dx = float(np.median(np.diff(t)))
    return savgol_filter(
        P_Torr * 133.322,
        window_length=window,
        polyorder=poly,
        deriv=1,
        delta=dx,
        mode="interp",
    )


# =============================================================================
# MAIN LOOP
# =============================================================================
def main():
    print(
        f"Carbon steel substrate Arrhenius fit: D_0={D_0_SUB:.3e}, "
        f"E_D={E_D_SUB:.4f} eV, Phi_0={PHI_0_SUB:.3e}, E_Phi={E_PHI_SUB:.4f} eV"
    )
    print(f"Tungsten barrier (Zhou D, Esteban S): D_0={D_0_W:.3e}, E_D={E_D_W:.4f}")
    print(f"Output dir: {FIGS_DIR}")

    summary = []
    for T_target in TEMPERATURES_K:
        print("\n" + "=" * 70)
        print(f"Target temperature: {T_target:.1f} K")
        run = closest_run(T_target)
        print(f"  closest experimental run: {run['name']} "
              f"(T={run['T']:.1f} K, P_up={run['P_up_Torr']:.1f} Torr)")

        exp = load_experiment(run["name"])
        # keep the full experiment, only chop a little of the pre-trigger tail
        mask = exp["t"] >= -50.0
        t_exp = exp["t"][mask]
        P_exp_Torr = exp["P_down_Torr"][mask]
        # subtract baseline (initial offset of the gauge before v3_open)
        baseline_mask = t_exp < 0.0
        if baseline_mask.any():
            P_exp_Torr = P_exp_Torr - np.mean(P_exp_Torr[baseline_mask])

        P_up_Pa = exp["P_up_Torr"] * 133.322
        T_sim_K = T_target  # use the requested temperature for the simulation
        t_exp_max = float(t_exp[-1])
        # match the FESTIM simulation length to the experiment duration
        final_time = max(t_exp_max, 60.0)

        print(f"  running FESTIM at T={T_sim_K:.1f} K, P_up={P_up_Pa:.2e} Pa, "
              f"t_final={final_time:.0f} s")
        t_sim, J_sim = run_festim(T_sim_K, P_up_Pa, final_time)
        P_sim_Torr = flux_to_pressure_rise_Torr(t_sim, J_sim, T_sim_K)

        # ---------- Plot 1: pressure rise ----------
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(t_exp, P_exp_Torr, color="black", lw=1.3, label="experiment")
        ax1.plot(t_sim, P_sim_Torr, color="C1", lw=2.0, label="FESTIM")
        ax1.set_xlabel("Time since upstream applied (s)")
        ax1.set_ylabel("Downstream pressure (Torr)")
        ax1.set_title(
            f"Downstream pressure rise — T_target={T_target:.0f} K\n"
            f"exp: {run['name']} (T={run['T']:.1f} K, P_up={run['P_up_Torr']:.0f} Torr)"
        )
        ax1.set_xlim(0, max(t_exp_max, float(t_sim[-1])))
        ax1.set_ylim(bottom=0)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig(FIGS_DIR / f"pressure_rise_T{int(T_target)}K.pdf",
                     bbox_inches="tight")
        plt.close(fig1)

        # ---------- Plot 2: permeability vs time ----------
        e_total = BARRIER_THICKNESS + SUBSTRATE_THICKNESS

        # FESTIM permeability (instantaneous)
        Phi_sim = J_sim * e_total / np.sqrt(P_up_Pa)

        # Experimental permeability via smoothed dP/dt
        t_exp_pos_mask = t_exp >= 0
        t_exp_pos = t_exp[t_exp_pos_mask]
        P_exp_pos = P_exp_Torr[t_exp_pos_mask]
        if len(t_exp_pos) > 200:
            window = min(155, (len(t_exp_pos) // 2) | 1)
            dPdt_Pa = smooth_dPdt(t_exp_pos, P_exp_pos, window=window)
            Phi_exp = permeability_from_dPdt(dPdt_Pa, P_up_Pa, T_sim_K, e_total)
        else:
            Phi_exp = np.full_like(t_exp_pos, np.nan)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(t_exp_pos, Phi_exp, color="black", lw=1.3, label="experiment")
        ax2.plot(t_sim, Phi_sim, color="C1", lw=2.0, label="FESTIM")
        ax2.set_xlabel("Time since upstream applied (s)")
        ax2.set_ylabel(r"Permeability  (atoms/(m·s·Pa$^{0.5}$))")
        ax2.set_title(
            f"Apparent permeability — T_target={T_target:.0f} K\n"
            f"exp: {run['name']}"
        )
        ax2.set_yscale("log")
        ax2.set_xlim(0, max(t_exp_max, float(t_sim[-1])))

        # y-axis floor: 2 decades below the smallest positive experimental value
        Phi_exp_pos = Phi_exp[np.isfinite(Phi_exp) & (Phi_exp > 0)]
        if Phi_exp_pos.size:
            phi_floor = float(Phi_exp_pos.min()) / 100.0
            phi_top = max(float(np.nanmax(Phi_sim)), float(Phi_exp_pos.max())) * 3
            ax2.set_ylim(phi_floor, phi_top)

        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(FIGS_DIR / f"permeability_T{int(T_target)}K.pdf",
                     bbox_inches="tight")
        plt.close(fig2)

        summary.append(
            {
                "T_target": T_target,
                "run": run["name"],
                "P_up_Torr": run["P_up_Torr"],
                "P_sim_final_Torr": float(P_sim_Torr[-1]),
                "Phi_sim_final": float(Phi_sim[-1]),
            }
        )

    print("\nSUMMARY")
    for s in summary:
        print(
            f"  T={s['T_target']:.0f} K | run={s['run']} | "
            f"P_up={s['P_up_Torr']:.0f} Torr | "
            f"P_sim_final={s['P_sim_final_Torr']:.3e} Torr | "
            f"Phi_sim_final={s['Phi_sim_final']:.3e}"
        )


if __name__ == "__main__":
    main()

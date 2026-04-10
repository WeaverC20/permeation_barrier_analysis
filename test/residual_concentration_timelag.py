"""
Residual Concentration Effect on GDP Timelag (No Trapping)

Shows that residual mobile hydrogen in tungsten (without trapping) has
negligible effect on the measured timelag in a GDP experiment.

Each case runs 3 phases in a single continuous FESTIM simulation:
  1. GDP Run 1 - reach steady-state permeation (P_up = 1e5 Pa)
  2. Bakeout - both sides evacuated (P = 0) for varying duration
  3. GDP Run 2 - measure timelag from this phase
"""

import sys
from pathlib import Path

# Allow importing from two_layer_numerical
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "two_layer_numerical"))

import festim as F
import numpy as np
import matplotlib.pyplot as plt
import ufl
from scipy import stats, integrate
from timelag_analysis_functions import StepsizeWithPostMilestoneControl

# =============================================================================
# Constants
# =============================================================================
k_B = 8.617333262145e-5  # eV/K

# Tungsten material properties
D_0 = 4.1e-7       # m^2/s
E_D = 0.39          # eV
K_S_0 = 1.87e24     # H/m^3/Pa^0.5
E_K_S = 1.04        # eV

# Simulation parameters
T = 700                     # K
substrate_thick = 1e-3      # m
P_run = 1.0e5               # Pa (upstream pressure during GDP runs)
run_length_phase1 = 1.25e4  # s (Phase 1: long enough for steady state)
run_length_phase3 = 4000    # s (Phase 2: 2nd GDP run for timelag measurement)

# Derived quantities
D_at_T = D_0 * np.exp(-E_D / (k_B * T))
S_at_T = K_S_0 * np.exp(-E_K_S / (k_B * T))
c_upstream = S_at_T * np.sqrt(P_run)
theoretical_timelag = substrate_thick**2 / (6 * D_at_T)

print(f"D({T}K) = {D_at_T:.3e} m^2/s")
print(f"S({T}K) = {S_at_T:.3e} H/m^3/Pa^0.5")
print(f"c_upstream = {c_upstream:.3e} H/m^3")
print(f"Theoretical timelag = {theoretical_timelag:.1f} s")
print(f"Diffusion time (L^2/D) = {substrate_thick**2 / D_at_T:.1f} s")
print()

# =============================================================================
# Bakeout cases
# =============================================================================
cases = {
    "Full bakeout (10000 s)":   10000,
    "Medium bakeout (1000 s)":  1000,
    "Short bakeout (200 s)":    200,
}

# =============================================================================
# Analytical residual concentration profile
# =============================================================================
def residual_profile(x, L, c0, D, t_bake, n_terms=200):
    """
    Residual concentration after bakeout from initial steady-state linear
    profile c(x) = c0 * (1 - x/L), with c=0 BCs on both sides.

    Uses Fourier sine series solution of the diffusion equation.
    """
    c = np.zeros_like(x, dtype=float)
    for n in range(1, n_terms + 1):
        # Fourier coefficient for f(x) = c0*(1 - x/L)
        Bn = 2 * c0 / (n * np.pi) * (-1) ** (n + 1)
        decay = np.exp(-D * (n * np.pi / L) ** 2 * t_bake)
        c += Bn * np.sin(n * np.pi * x / L) * decay
    return c


# =============================================================================
# Timelag calculation
# =============================================================================
def fit_linear_asymptote(t, Q):
    slope, intercept, r_value, _, _ = stats.linregress(t, Q)
    return slope, intercept, r_value**2


def calculate_timelag(t_seg, flux_seg, thickness):
    """Calculate timelag from a GDP run flux segment."""
    t_shifted = t_seg - t_seg[0]
    flux_abs = np.abs(flux_seg)
    Q = integrate.cumulative_trapezoid(flux_abs, t_shifted, initial=0)

    n_points = len(t_shifted)
    steady_start = int(0.7 * n_points)

    slope, intercept, r_sq = fit_linear_asymptote(
        t_shifted[steady_start:], Q[steady_start:]
    )

    t_lag = -intercept / slope if abs(slope) > 1e-20 else np.nan
    D_eff = thickness**2 / (6 * t_lag) if (t_lag > 0 and np.isfinite(t_lag)) else np.nan

    return t_lag, D_eff, r_sq, t_shifted, flux_abs, Q, slope, intercept


# =============================================================================
# Run a single case
# =============================================================================
def run_case(bakeout_time):
    """Run a 3-phase GDP simulation and return Phase 3 results."""

    t_phase1_end = run_length_phase1
    t_phase2_end = run_length_phase1 + bakeout_time
    final_time = run_length_phase1 + bakeout_time + run_length_phase3
    milestones = [t_phase1_end, t_phase2_end]

    # --- Model setup ---
    my_model = F.HydrogenTransportProblem()

    # Mesh (power-law graded, 100 points)
    N = 100
    s = np.linspace(0, 1, N)
    vertices = substrate_thick * s**2.0
    my_model.mesh = F.Mesh1D(vertices)

    # Material
    tungsten = F.Material(D_0=D_0, E_D=E_D, K_S_0=K_S_0, E_K_S=E_K_S)

    # Species (no trapping)
    H = F.Species("H")
    my_model.species = [H]

    # Subdomains
    volume = F.VolumeSubdomain1D(id=1, borders=(0, substrate_thick), material=tungsten)
    left = F.SurfaceSubdomain1D(id=2, x=0)
    right = F.SurfaceSubdomain1D(id=3, x=substrate_thick)
    my_model.subdomains = [left, volume, right]
    my_model.surface_to_volume = {left: volume, right: volume}

    # Temperature
    my_model.temperature = T

    # Pressure function (UFL conditional)
    def P_up_func(t):
        return ufl.conditional(
            ufl.le(t, t_phase1_end),
            P_run,
            ufl.conditional(
                ufl.le(t, t_phase2_end),
                0.0,
                P_run,
            ),
        )

    # Boundary conditions
    left_bc = F.SievertsBC(
        subdomain=left,
        S_0=tungsten.K_S_0,
        E_S=tungsten.E_K_S,
        pressure=P_up_func,
        species=H,
    )
    right_bc = F.FixedConcentrationBC(subdomain=right, value=0, species=H)
    my_model.boundary_conditions = [left_bc, right_bc]

    # Exports
    permeation_flux = F.SurfaceFlux(field=H, surface=right)
    my_model.exports = [permeation_flux]

    # Settings with milestone control
    my_model.settings = F.Settings(
        atol=1e10,
        rtol=1e-08,
        max_iterations=1000,
        final_time=final_time,
    )
    my_model.settings.stepsize = StepsizeWithPostMilestoneControl(
        initial_value=1,
        growth_factor=1.05,
        cutback_factor=0.5,
        max_stepsize=200,
        target_nb_iterations=30,
        milestones=milestones,
        pre_milestone_duration=100,
        post_milestone_duration=500,
    )

    # Run
    my_model.initialise()
    my_model.run()

    # Extract Phase 3 (second GDP run)
    times = np.asarray(permeation_flux.t)
    flux = np.asarray(permeation_flux.data)

    mask = times > t_phase2_end
    t_phase3 = times[mask]
    f_phase3 = flux[mask]

    return t_phase3, f_phase3


# =============================================================================
# Run all cases
# =============================================================================
results = {}
for label, bakeout_time in cases.items():
    print(f"Running: {label} ...")
    t_seg, f_seg = run_case(bakeout_time)
    t_lag, D_eff, r_sq, t_shifted, flux_abs, Q, slope, intercept = calculate_timelag(
        t_seg, f_seg, substrate_thick
    )
    results[label] = {
        "bakeout_time": bakeout_time,
        "t_lag": t_lag,
        "D_eff": D_eff,
        "r_sq": r_sq,
        "t_shifted": t_shifted,
        "flux_abs": flux_abs,
        "Q": Q,
        "slope": slope,
        "intercept": intercept,
    }
    print(f"  timelag = {t_lag:.1f} s, D_eff = {D_eff:.3e} m^2/s, R^2 = {r_sq:.6f}")

# =============================================================================
# Summary table
# =============================================================================
print("\n" + "=" * 80)
print(f"{'Case':<30s} | {'Bakeout (s)':>11s} | {'t_lag (s)':>10s} | {'D_eff (m^2/s)':>14s} | {'% diff':>7s}")
print("-" * 80)

ref_tlag = results["Full bakeout (10000 s)"]["t_lag"]
for label, res in results.items():
    pct = (res["t_lag"] - ref_tlag) / ref_tlag * 100
    print(
        f"{label:<30s} | {res['bakeout_time']:>11d} | {res['t_lag']:>10.1f} | {res['D_eff']:>14.3e} | {pct:>+7.2f}%"
    )
print(f"{'Theoretical (L^2/6D)':<30s} | {'--':>11s} | {theoretical_timelag:>10.1f} | {D_at_T:>14.3e} |    --")
print("=" * 80)

# =============================================================================
# Plot
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = ["#1b9e77", "#d95f02", "#7570b3"]  # colorblind-friendly

# --- Top-left: Concentration profiles (steady-state + residuals) ---
ax = axes[0, 0]
x = np.linspace(0, substrate_thick, 500)

# Steady-state linear profile
c_ss = c_upstream * (1 - x / substrate_thick)
ax.plot(x * 1e3, c_ss, color="black", linewidth=2, linestyle="--", label="Steady state")

# Residual profiles
for i, (label, bakeout_time) in enumerate(cases.items()):
    c_residual = residual_profile(x, substrate_thick, c_upstream, D_at_T, bakeout_time)
    ax.plot(x * 1e3, c_residual, color=colors[i], linewidth=2, label=label)

ax.set_xlabel("Position (mm)")
ax.set_ylabel("Concentration (H/m$^3$)")
ax.set_title("Concentration Profiles at Start of 2nd GDP Run")
ax.legend()
ax.set_xlim(0, substrate_thick * 1e3)

# --- Top-right: Flux vs time for 2nd GDP run ---
ax = axes[0, 1]
for i, (label, res) in enumerate(results.items()):
    ax.plot(res["t_shifted"], res["flux_abs"], color=colors[i], linewidth=1.5, label=label)

ax.set_xlabel("Time since start of 2nd GDP run (s)")
ax.set_ylabel("|Flux| (H/m$^2$/s)")
ax.set_title("Downstream Flux During 2nd GDP Run")
ax.legend()

# --- Bottom-left: Cumulative Q with asymptote fits ---
ax = axes[1, 0]
for i, (label, res) in enumerate(results.items()):
    ax.plot(res["t_shifted"], res["Q"], color=colors[i], linewidth=1.5, label=label)
    # Plot asymptote
    t_fit = np.array([0, res["t_shifted"][-1]])
    Q_fit = res["slope"] * t_fit + res["intercept"]
    ax.plot(t_fit, Q_fit, "--", color=colors[i], linewidth=1, alpha=0.7)
    # Mark timelag
    ax.axvline(res["t_lag"], color=colors[i], linestyle=":", linewidth=1, alpha=0.7)

ax.set_xlabel("Time since start of 2nd GDP run (s)")
ax.set_ylabel("Cumulative permeated H (H/m$^2$)")
ax.set_title("Cumulative Q(t) with Linear Asymptotes")
ax.legend()

# --- Bottom-right: Bar chart of timelags ---
ax = axes[1, 1]
labels_short = [l.split("(")[0].strip() for l in cases.keys()]
timelags = [results[l]["t_lag"] for l in cases.keys()]
bars = ax.bar(labels_short, timelags, color=colors, edgecolor="black", linewidth=0.8)

# Annotate % diff
for i, (label, res) in enumerate(results.items()):
    pct = (res["t_lag"] - ref_tlag) / ref_tlag * 100
    ax.text(
        i, res["t_lag"] + max(timelags) * 0.01,
        f"{res['t_lag']:.1f} s\n({pct:+.2f}%)",
        ha="center", va="bottom", fontsize=9,
    )

ax.axhline(theoretical_timelag, color="black", linestyle="--", linewidth=1, label=f"Theoretical ({theoretical_timelag:.1f} s)")
ax.set_ylabel("Timelag (s)")
ax.set_title("Timelag Comparison")
ax.legend()

fig.suptitle(
    f"Effect of Residual Concentration on GDP Timelag\n"
    f"Tungsten, {T} K, no trapping, L = {substrate_thick*1e3:.0f} mm",
    fontsize=14, fontweight="bold",
)
fig.tight_layout()

output_path = Path(__file__).resolve().parent / "residual_concentration_timelag.pdf"
fig.savefig(output_path, dpi=200)
print(f"\nFigure saved to: {output_path}")
plt.show()

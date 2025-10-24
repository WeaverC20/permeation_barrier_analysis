import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Iterable, Optional
import csv
import math

param_keys = [
    "D_0_barrier", "E_D_barrier", "S_0_barrier", "E_S_barrier",
    "K_r_0_barrier", "E_K_r_barrier",
    "D_0_substrate", "E_D_substrate", "S_0_substrate", "E_S_substrate",
    "K_r_0_substrate", "E_K_r_substrate",
]

fieldnames = (
    ["run", "barrier_thickness", "substrate_thickness", "T", "P_up",
     "C1", "Cm1", "Cm2", "C2", "W", "R", "flux"]
    + param_keys
)

def determine_regime(C1, Cm1, Cm2, C2, tolerance=0.05):
    """
    determines the regime based on concentration profile at steady state
    if difference in concentration across layer less than "tolerance input" * C1, then it is surface limited
    """
    if abs(C1 - Cm1) / C1 < tolerance:
        regime_barrier = "limited"
    else:
        regime_barrier = "limiting"

    if abs(Cm2 - C2) / C1 < tolerance:
        regime_substrate = "limited"
    else:
        regime_substrate = "limiting"

    if regime_barrier == "limited" and regime_substrate == "limited":
        regime_system = "surface-limited"
    elif regime_barrier == "limiting" and regime_substrate == "limiting":
        regime_system = "diffusion-limited"
    else:
        regime_system = "mixed"

    return regime_barrier, regime_substrate, regime_system

def generate_params_list(base_params, S_0_limits, D_0_limits, n_points_S, n_points_D):
    """
    takes in the limits for S_0 and D_0 and generates a list of parameter sets
    that spans the space of S_0 and D_0 values
    uses a scale with more points in the middle of the range

    outputs a list of dictionaries of size n_points_R^2 * n_points_W^2
    """
    S_0_values = np.logspace(
        np.log10(S_0_limits[0]),
        np.log10(S_0_limits[1]),
        n_points_S
    )
    D_0_values = np.logspace(
        np.log10(D_0_limits[0]),
        np.log10(D_0_limits[1]),
        n_points_D
    )

    param_list = []
    for S_0_barrier in S_0_values:
        for S_0_substrate in S_0_values:
            for D_0_barrier in D_0_values:
                for D_0_substrate in D_0_values:
                    params = base_params.copy()
                    params["D_0_barrier"] = D_0_barrier
                    params["S_0_barrier"] = S_0_barrier
                    params["D_0_substrate"] = D_0_substrate
                    params["S_0_substrate"] = S_0_substrate
                    param_list.append(params)

    return param_list

def generate_params_list_P_varying(base_params, S_0_limits, D_0_limits, n_points_S, n_points_D, P_list):
    """
    takes in the limits for S_0 and D_0 and generates a list of parameter sets
    that spans the space of S_0 and D_0 values
    uses a scale with more points in the middle of the range

    outputs a list of dictionaries of size n_points_R^2 * n_points_W^2
    """
    S_0_values = np.logspace(
        np.log10(S_0_limits[0]),
        np.log10(S_0_limits[1]),
        n_points_S
    )
    D_0_values = np.logspace(
        np.log10(D_0_limits[0]),
        np.log10(D_0_limits[1]),
        n_points_D
    )

    param_list = []
    for S_0_barrier in S_0_values:
        for S_0_substrate in S_0_values:
            for D_0_barrier in D_0_values:
                for D_0_substrate in D_0_values:
                    for P in P_list:
                        params = base_params.copy()
                        params["D_0_barrier"] = D_0_barrier
                        params["S_0_barrier"] = S_0_barrier
                        params["D_0_substrate"] = D_0_substrate
                        params["S_0_substrate"] = S_0_substrate
                        param_list.append(params)

    return param_list

def _to_float(x):
    """Safe float conversion (handles '1e-11', '', None). Returns np.nan on failure."""
    try:
        return float(x)
    except Exception:
        return float("nan")


def analyze_csv_and_plot(
    csv_path: str | Path,
    tolerance: float = 0.05,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> list[dict]:
    """
    Reads the CSV, computes regime for each row, and plots W vs R on log scales.
    Returns a list of annotated rows (dicts) with added keys:
        'regime_barrier', 'regime_substrate', 'regime_system'
    """
    csv_path = Path(csv_path)
    results: list[dict] = []

    # Read CSV as dicts
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i in (0, 1):
                continue  # skip first and second row
            # Pull needed fields (robustly to float)
            C1  = _to_float(row.get("C1"))
            Cm1 = _to_float(row.get("Cm1"))
            Cm2 = _to_float(row.get("Cm2"))
            C2  = _to_float(row.get("C2"))
            W   = _to_float(row.get("W"))
            R   = _to_float(row.get("R"))

            # skip rows with bad/zero/negative data for log plot
            if not all(map(np.isfinite, [C1, Cm1, Cm2, C2, W, R])) or W <= 0 or R <= 0:
                continue

            # Apply regime rule
            _, _, regime_system = determine_regime(C1, Cm1, Cm2, C2, tolerance=tolerance)

            # stash annotated row
            row_out = dict(row)
            row_out["regime_system"] = regime_system
            results.append(row_out)

    # If nothing valid, exit early
    if not results:
        print("No valid rows to plot (check W/R > 0 and numeric fields).")
        return results

    # Prepare plotting data grouped by regime
    regimes = ["surface-limited", "diffusion-limited", "mixed"]
    colors = {
        "surface-limited": "#1953f2",   # blue
        "diffusion-limited": "#f7f702", # red
        "mixed": "#f7340c",             # green
    }

    # Build per-regime lists
    data_by_regime: dict[str, list[tuple[float, float]]] = {k: [] for k in regimes}
    for r in results:
        regime = r["regime_system"]
        W_val = _to_float(r["W"])
        R_val = _to_float(r["R"])
        if np.isfinite(W_val) and np.isfinite(R_val) and W_val > 0 and R_val > 0:
            if regime not in data_by_regime:
                data_by_regime[regime] = []
            data_by_regime[regime].append((W_val, R_val))

    # Plot
    plt.figure(figsize=(7, 6))
    for regime in regimes:
        pts = data_by_regime.get(regime, [])
        if not pts:
            continue
        W_vals, R_vals = zip(*pts)
        plt.scatter(W_vals, R_vals, label=regime, s=80, alpha=0.85, edgecolors="none", c=colors[regime])

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("W")
    plt.ylabel("R")
    plt.title("W vs R by Regime")
    plt.legend(title="Regime", loc="best")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return results
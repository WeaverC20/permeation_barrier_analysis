import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pandas as pd
import csv
from uncertainties import ufloat
import re

def fit_linear_asymptote(x, y, tail_frac=0.25, tol=0.08):
    """Fit a linear asymptote to the final section of (x, y).

    Returns (slope, intercept, start_index_used).
    - tail_frac: fraction of the end to examine for a stable gradient.
    - tol: relative tolerance for gradient stability (e.g. 0.08 = 8%).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size < 5:
        raise ValueError("need at least 5 points to fit an asymptote")

    g = np.gradient(y, x)
    n = len(x)
    tail_start = max(0, int(n * (1 - tail_frac)))
    tail_g = g[tail_start:]

    med = np.median(tail_g)
    if med == 0:
        med = np.mean(tail_g) or 1e-12

    rel_ok = np.abs((tail_g - med) / (med if med != 0 else 1e-12)) <= tol

    use_rel_idx = None
    for i in range(len(rel_ok)):
        if rel_ok[i:].all():
            use_rel_idx = tail_start + i
            break
    if use_rel_idx is None:
        use_rel_idx = tail_start

    Xfit = x[use_rel_idx:]
    Yfit = y[use_rel_idx:]
    slope, intercept = np.polyfit(Xfit, Yfit, 1)
    return slope, intercept, use_rel_idx

def load_downstream_data(filepath):
    """
    Loads CSV with a datetime RealTimestamp column and returns:
    - time_s : seconds since start (numpy array)
    - V_Baratron : downstream baratron voltage (numpy array)
    - V_Wasp : wasp voltage (numpy array)
    - df : cleaned DataFrame with Python-safe column names
    """
    # Read CSV
    df = pd.read_csv(filepath)

    # Parse datetimes
    df["RealTimestamp"] = pd.to_datetime(df["RealTimestamp"], errors="coerce")

    # Time since start (0 = first row)
    t0 = df["RealTimestamp"].iloc[0]
    time_s = (df["RealTimestamp"] - t0).dt.total_seconds().to_numpy()

    # Rename columns to be Python-friendly
    df = df.rename(columns=lambda x: x.strip().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_per_"))

    return time_s, df

def voltage_to_torr_wasp_downstream(voltage):
    """Convert Wasp voltage to pressure in Pa."""
    # Calibration from manual
    indicated_pressure = 10**((voltage - 5.5)/0.5)

    return np.array([P if P >=1e-5 else P * 2.4 for P in indicated_pressure])

def voltage_to_torr_baratron(voltage):
    """Convert Wasp voltage to pressure in Torr."""
    # Calibration from manual
    return np.array(voltage * 100)

def average_pressure_after_increase(time, pressure, window=5, slope_threshold=1e-3):
    """
    Detects when the pressure stabilizes after a sudden increase and 
    returns the average pressure after that time in torr.

    Parameters
    ----------
    time : array-like
        Time values (seconds).
    pressure : array-like
        Pressure values corresponding to time.
    window : int, optional
        Number of points to use for local slope estimation (default=5).
    slope_threshold : float, optional
        Threshold for determining when slope is "flat" (default=1e-3).

    Returns
    -------
    avg_pressure : float
        Average pressure after the increase.
    t_start : float
        Detected time when pressure flattens.
    """

    time = np.asarray(time)
    pressure = np.asarray(pressure)

    # Estimate slope using central differences
    slopes = np.gradient(pressure, time)

    # Smooth slope with rolling average
    smooth_slopes = np.convolve(slopes, np.ones(window)/window, mode='same')

    # Find first time slope falls below threshold after the jump
    for i in range(len(smooth_slopes)):
        if abs(smooth_slopes[i]) < slope_threshold and time[i] > min(time) + 5:
            # treat this as the "settled" point
            settled_index = i
            break
    else:
        settled_index = int(0.5*len(time))  # fallback: halfway

    # Compute average after that point
    avg_pressure = np.mean(pressure[settled_index:])
    # t_start = time[settled_index]

    return avg_pressure

def write_run_to_csv(csv_filename, run_name, material, temperature_c, diffusivity, permeability):
    """
    Writes run info into the CSV file.
    If the run already exists (based on run_name), its row is updated.
    Otherwise, a new row is appended.
    """
    rows = []
    headers = ["Run Name", "Material", "Temperature (K)", "Diffusivity", "Permeability"]
    file_exists = os.path.isfile(csv_filename)

    # Load existing rows if file exists
    if file_exists:
        with open(csv_filename, mode="r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    # Check if run_name already exists and update, otherwise append
    updated = False
    for row in rows:
        if row["Run Name"] == run_name:
            row["Material"] = material
            row["Temperature (K)"] = temperature_c
            row["Diffusivity"] = diffusivity
            row["Permeability"] = permeability
            updated = True
            break

    if not updated:
        rows.append({
            "Run Name": run_name,
            "Material": material,
            "Temperature (K)": temperature_c,
            "Diffusivity": diffusivity,
            "Permeability": permeability
        })

    # Rewrite the file with headers + updated rows
    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def parse_ufloat(value):
    """
    Parse a value like '(8.9+/-1.0)e+08' or '3.4e-12' into a ufloat or float.
    """
    if isinstance(value, str):
        # Match (val+/-err)e+exp or similar formats
        match = re.match(r"\(?([0-9.\-eE]+)\+/-([0-9.\-eE]+)\)?(?:e([+\-]?[0-9]+))?", value.strip())
        if match:
            val = float(match.group(1))
            err = float(match.group(2))
            exp = int(match.group(3)) if match.group(3) else 0
            return ufloat(val * 10**exp, err * 10**exp)
        else:
            # Try a simple float fallback
            try:
                return float(value)
            except ValueError:
                return None
    return value  # already numeric
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pandas as pd

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

def voltage_to_Pa_wasp(voltage):
    """Convert Wasp voltage to pressure in Pa."""
    # Calibration from manual
    indicated_pressure = 10**((voltage - 5.5)/0.5)

    return np[P if P >=1e-5 else P * 2.4 for P in indicated_pressure]
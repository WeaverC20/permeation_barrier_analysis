import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pandas as pd
import csv
from uncertainties import ufloat
import re

def fit_linear_asymptote(x, y, tail_frac=0.25):
    """Fit a linear asymptote to the final section of (x, y).

    Returns (slope, intercept, start_index_used).
    - tail_frac: fraction of the end to examine for a stable gradient.
    - tol: relative tolerance for gradient stability (e.g. 0.08 = 8%).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size < 5:
        raise ValueError("need at least 5 points to fit an asymptote")

    n = len(x)
    tail_start = max(0, int(n * (1 - tail_frac)))

    Xfit = x[tail_start:]
    Yfit = y[tail_start:]
    slope, intercept = np.polyfit(Xfit, Yfit, 1)
    return slope, intercept, tail_start

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

    def H_correction_high(p_i):
        """
        pressure correction valid for 7.6*10^-3 < pressure < 76 torr
        """
        a = 0.3391937 
        b = 0.8666103
        c = 0.1400703
        d = 0.0460218
        e = 0.0001714538 
        f = 0.0002287221

        return a + b*p_i + c*p_i**2 + d*p_i**3 + e*p_i**4 + f*p_i**5

    def H_correction_low(indicated_pressure):
        """
        pressure correction valid for 7.6*10^-7 < pressure < 7.6*10^-3 torr
        """
        return indicated_pressure * 2.4

    return np.array([H_correction_low(P) if P <1e-1 else H_correction_high(P) for P in indicated_pressure])

def voltage_to_torr_baratron_upstream(voltage):
    """Convert Wasp voltage to pressure in Torr."""
    # Calibration from manual
    return np.array(voltage * 100)

def voltage_to_torr_baratron_downstream(voltage):
    """Convert Wasp voltage to pressure in Torr."""
    # Calibration from manual
    return np.array(voltage * 100 / 1000)

def average_pressure_after_increase(time, pressure, window=5, slope_threshold=1e-3, tail_frac=0.25):
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

    tail_start = len(pressure) - int((1-tail_frac) * len(pressure))

    # Compute average after that point
    avg_pressure = np.mean(pressure[tail_start:])
    # t_start = time[settled_index]

    return avg_pressure

def pressure_timelag(pressure, t, tau_l):
    """
    pressure used in calculation assuming timelag between hydrogen entering chamber and pressure reading
    """
    



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

def volts_to_temp_constants(mv: float) -> tuple[float, ...]:
    """
    Select the appropriate NIST ITS-90 polynomial coefficients for converting
    Type K thermocouple voltage (in millivolts) to temperature (°C).

    The valid voltage range is -5.891 mV to 54.886 mV.

    args:
        mv: Thermocouple voltage in millivolts.

    returns:
        tuple of float: Polynomial coefficients for the voltage-to-temperature conversion.

    raises:
        ValueError: If the input voltage is out of the valid range.
    """
    # Use a small tolerance for floating-point comparison
    if mv < -5.892 or mv > 54.887:
        raise ValueError("Voltage out of valid Type K range (-5.891 to 54.886 mV).")
    if mv < 0:
        # Range: -5.891 mV to 0 mV
        return (
            0.0e0,
            2.5173462e1,
            -1.1662878e0,
            -1.0833638e0,
            -8.977354e-1,
            -3.7342377e-1,
            -8.6632643e-2,
            -1.0450598e-2,
            -5.1920577e-4,
        )
    elif mv < 20.644:
        # Range: 0 mV to 20.644 mV
        return (
            0.0e0,
            2.508355e1,
            7.860106e-2,
            -2.503131e-1,
            8.31527e-2,
            -1.228034e-2,
            9.804036e-4,
            -4.41303e-5,
            1.057734e-6,
            -1.052755e-8,
        )
    else:
        # Range: 20.644 mV to 54.886 mV
        return (
            -1.318058e2,
            4.830222e1,
            -1.646031e0,
            5.464731e-2,
            -9.650715e-4,
            8.802193e-6,
            -3.11081e-8,
        )

def evaluate_poly(coeffs: list[float] | tuple[float], x: float) -> float:
    """ "
    Evaluate a polynomial at x given the list of coefficients.

    The polynomial is:
        P(x) = a0 + a1*x + a2*x^2 + ... + an*x^n
    where coeffs = [a0, a1, ..., an]

    args:
        coeffs:Polynomial coefficients ordered by ascending power.
        x: The value at which to evaluate the polynomial.

    returns;
        float: The evaluated polynomial result.
    """
    return sum(a * x**i for i, a in enumerate(coeffs))

def mv_to_temp_c(mv):
    """
    Convert Type K thermocouple voltage (mV) to temperature (°C) using
    NIST ITS-90 polynomial approximations.

    args:
        mv: Thermocouple voltage in millivolts.

    returns:
        float: Temperature in degrees Celsius.
    """
    coeffs = volts_to_temp_constants(mv)
    return evaluate_poly(coeffs, mv)

def voltage_to_temp_typeK(voltage_mV):
    return np.array([mv_to_temp_c(mv) for mv in voltage_mV])
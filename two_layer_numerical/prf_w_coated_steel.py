import festim as F
import numpy as np
import h_transport_materials as htm
import csv
import re
import os

from festim2_solve_functions import create_mesh, assign_materials_and_domains, set_exports

# --- constants ---
k_B = 8.617333262e-5  # eV/K
T = 540  # K
P_up = 100  # Pa
barrier_thickness = 1e-7  # m (100 nm)
substrate_thickness = 6.5e-4  # m (650 micron)

# --- tungsten properties from htm (Frauenfelder) ---
D_W = htm.diffusivities.filter(material="tungsten").filter(isotope="h").filter(author="zhou")[0]
S_W = htm.solubilities.filter(material="tungsten").filter(isotope="h").filter(author="esteban")[0]

D_0_barrier = D_W.pre_exp.magnitude
E_D_barrier = D_W.act_energy.magnitude
S_0_barrier = S_W.pre_exp.magnitude
E_S_barrier = S_W.act_energy.magnitude

# --- carbon steel properties from Arrhenius fit of experimental data ---
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "ShieldRunsAnalysis", "results", "figs", "fresh_steel_diffusivities.csv")

T_data = []
D_data = []
Phi_data = []

with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        T_data.append(float(row["Temperature (K)"]))
        D_data.append(float(row["Diffusivity"]))

        # parse permeability string like "(5.9+/-1.2)e+12"
        perm_str = row["Permeability"].strip()
        match = re.match(r"\(?([0-9.\-eE]+)\+/-([0-9.\-eE]+)\)?(?:e([+\-]?[0-9]+))?", perm_str)
        if match:
            val = float(match.group(1))
            exp = int(match.group(3)) if match.group(3) else 0
            Phi_data.append(val * 10**exp)
        else:
            Phi_data.append(float(perm_str))

T_arr = np.array(T_data)
D_arr = np.array(D_data)
Phi_arr = np.array(Phi_data)

# Arrhenius fit: ln(y) vs 1/T
m_D, b_D = np.polyfit(1 / T_arr, np.log(D_arr), 1)
D_0_substrate = np.exp(b_D)
E_D_substrate = -m_D * k_B  # eV

m_Phi, b_Phi = np.polyfit(1 / T_arr, np.log(Phi_arr), 1)
Phi_0 = np.exp(b_Phi)
E_Phi = -m_Phi * k_B  # eV

# derive solubility: S = Phi / D
S_0_substrate = Phi_0 / D_0_substrate
E_S_substrate = E_Phi - E_D_substrate  # eV

# --- compute bare-steel flux from permeability Arrhenius fit ---
Phi_T = Phi_0 * np.exp(-E_Phi / (k_B * T))
J_bare = Phi_T * np.sqrt(P_up) / substrate_thickness

# --- run two-layer FESTIM simulation (W barrier + carbon steel) ---
params = {
    "D_0_barrier": D_0_barrier,
    "E_D_barrier": E_D_barrier,
    "S_0_barrier": S_0_barrier,
    "E_S_barrier": E_S_barrier,
    "D_0_substrate": D_0_substrate,
    "E_D_substrate": E_D_substrate,
    "S_0_substrate": S_0_substrate,
    "E_S_substrate": E_S_substrate,
}

my_model = F.HydrogenTransportProblemDiscontinuous()
create_mesh(my_model, barrier_thickness, substrate_thickness)
barrier, substrate, left, right, H = assign_materials_and_domains(
    my_model, params, barrier_thickness, substrate_thickness, T=T, P_up=P_up
)

my_model.settings = F.Settings(
    atol=1e-0,
    rtol=1e-10,
    transient=False,
)

barrier_export, substrate_export, flux_right = set_exports(
    my_model, barrier, substrate, left, right, H, results_folder="results/prf_w_coated"
)

my_model.initialise()
my_model.run()

flux_with_barrier = flux_right.data[0]

# --- compute PRF ---
PRF = abs(J_bare) / abs(flux_with_barrier)

# --- print results ---
print("=" * 60)
print("Tungsten barrier (Frauenfelder, htm)")
print(f"  D_0 = {D_0_barrier:.3e} m²/s,  E_D = {E_D_barrier:.4f} eV")
print(f"  S_0 = {S_0_barrier:.3e} H/m³/Pa⁰·⁵,  E_S = {E_S_barrier:.4f} eV")

print("\nCarbon steel substrate (Arrhenius fit of experimental data)")
print(f"  D_0 = {D_0_substrate:.3e} m²/s,  E_D = {E_D_substrate:.4f} eV")
print(f"  Φ_0 = {Phi_0:.3e} H/(m·s·Pa⁰·⁵),  E_Φ = {E_Phi:.4f} eV")
print(f"  S_0 = {S_0_substrate:.3e} H/m³/Pa⁰·⁵,  E_S = {E_S_substrate:.4f} eV")

print(f"\nConditions: T = {T} K, P_up = {P_up} Pa")
print(f"  barrier = {barrier_thickness*1e9:.0f} nm W, substrate = {substrate_thickness*1e6:.0f} μm steel")

D_W_T = D_0_barrier * np.exp(-E_D_barrier / (k_B * T))
S_W_T = S_0_barrier * np.exp(-E_S_barrier / (k_B * T))
D_S_T = D_0_substrate * np.exp(-E_D_substrate / (k_B * T))
S_S_T = S_0_substrate * np.exp(-E_S_substrate / (k_B * T))

print(f"\nEvaluated at {T} K:")
print(f"  W:  D = {D_W_T:.3e} m²/s,  S = {S_W_T:.3e} H/m³/Pa⁰·⁵")
print(f"  Steel: D = {D_S_T:.3e} m²/s,  S = {S_S_T:.3e} H/m³/Pa⁰·⁵")
print(f"  Φ(steel) = {Phi_T:.3e} H/(m·s·Pa⁰·⁵)")

print(f"\nBare-steel flux (Arrhenius):  {J_bare:.3e} H/m²/s")
print(f"With-barrier flux (FESTIM):   {flux_with_barrier:.3e} H/m²/s")
print(f"\nPRF = {PRF:.2f}")
print("=" * 60)

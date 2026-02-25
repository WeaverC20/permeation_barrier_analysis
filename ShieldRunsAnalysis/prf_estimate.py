import numpy as np
import h_transport_materials as htm

# --- parameters ---
T = 540  # K
e_barrier = 8e-7  # m
e_substrate = 6.5e-4  # m

# --- material properties from htm ---
D_b = htm.diffusivities.filter(material="tungsten").filter(isotope="h").filter(author="frauenfelder")[0]
S_b = htm.solubilities.filter(material="tungsten").filter(isotope="h").filter(author="frauenfelder")[0]

D_s = htm.diffusivities.filter(material="316l_steel").filter(isotope="h").filter(author="reiter")[0]
S_s = htm.solubilities.filter(material="316l_steel").filter(isotope="h").filter(author="reiter")[0]

# --- evaluate at temperature ---
k_B = 8.617333262e-5  # eV/K

D_barrier = D_b.pre_exp.magnitude * np.exp(-D_b.act_energy.magnitude / (k_B * T))
S_barrier = S_b.pre_exp.magnitude * np.exp(-S_b.act_energy.magnitude / (k_B * T))
D_substrate = D_s.pre_exp.magnitude * np.exp(-D_s.act_energy.magnitude / (k_B * T))
S_substrate = S_s.pre_exp.magnitude * np.exp(-S_s.act_energy.magnitude / (k_B * T))

# --- PRF = 1 + 2 * alpha * beta * gamma ---
alpha = D_substrate / D_barrier
beta = S_substrate / S_barrier
gamma = e_barrier / e_substrate

PRF = 1 + 2 * alpha * beta * gamma

print(f"T = {T} K")
print(f"alpha (D_sub/D_bar) = {alpha:.4f}")
print(f"beta  (S_sub/S_bar) = {beta:.4e}")
print(f"gamma (e_bar/e_sub) = {gamma:.6f}")
print(f"PRF = {PRF:.4f}")
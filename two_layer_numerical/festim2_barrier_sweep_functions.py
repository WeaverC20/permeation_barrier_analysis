param_keys = [
    "D_0_barrier", "E_D_barrier", "S_0_barrier", "E_S_barrier",
    "K_r_0_barrier", "E_K_r_barrier",
    "D_0_substrate", "E_D_substrate", "S_0_substrate", "E_S_substrate",
    "K_r_0_substrate", "E_K_r_substrate",
]

fieldnames = (
    ["run", "barrier_thickness", "substrate_thickness", "T", "P_up",
     "C1", "Cm1", "Cm2", "C2", "W", "R"]
    + param_keys
)


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
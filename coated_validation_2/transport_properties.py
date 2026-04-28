"""coated_validation_2/transport_properties.py

Transport-property pack used by every script in this folder.  All values are
the ones we converged on in coated_validation/ -- see findings.md there.

  W coating:     Heinola 2010 (D), Esteban 2001 (K_s)
  Carbon steel:  empirical Arrhenius fit of the bare-CS SHIELD CSV
                 (`steel_diffusivities.csv`), split into D / K_s
  Steel K_r:     Eurofer97 sigma*k2 fit (analogous low-C/RAFM steel)

K_r for tungsten is intentionally NOT hard-coded here -- step1_kr_fit.py runs
the FESTIM 2 SurfaceReaction transient on run 03.19's pressure trace and
writes the fitted (K_r_0, E_K_r) pair to fitted_kr.json.  Steps 2 and 3 read
that file.
"""

from __future__ import annotations

import numpy as np

K_B_EV = 8.617_333_262e-5    # eV / K
N_A    = 6.022_140_76e23     # mol^-1

# --- material library -------------------------------------------------------

MATERIALS: dict[str, dict] = {
    "tungsten": {
        # Heinola D + Esteban K_s -- both H isotope.
        "D0":  5.2e-8,        # m^2 / s
        "E_D": 0.21,          # eV
        # K_s_0 stored in mol/m^3/Pa^0.5 = (1.7464e22 particle/m^3/Pa^0.5) / N_A
        "Ks0": 1.7464e22 / N_A,
        "E_S": 0.2788,
        "label": "tungsten (Heinola D + Esteban K_s)",
    },
    "carbon_steel": {
        # Two-Arrhenius split of the bare carbon-steel SHIELD CSV.
        "D0":   1.397e-6,
        "E_D":  0.3841,
        "Ks0":  8.28e-3,
        "E_S": -0.0616,
        # Recombination Arrhenius -- Eurofer97 sigma*k2 fit (analogous).
        "Kr0":  8.91e-2,      # m^4 / mol / s
        "E_Kr": 0.5308,
        "label": "low-carbon steel (SHIELD CSV fit + Eurofer97 K_r)",
    },
}


def D(material: str, T):
    p = MATERIALS[material]
    return p["D0"] * np.exp(-p["E_D"] / (K_B_EV * np.asarray(T, dtype=float)))


def Ks(material: str, T):
    p = MATERIALS[material]
    return p["Ks0"] * np.exp(-p["E_S"] / (K_B_EV * np.asarray(T, dtype=float)))


def Phi(material: str, T):
    return D(material, T) * Ks(material, T)


def Phi_atomH(material: str, T):
    return Phi(material, T) * N_A


def Kr_steel(T):
    """Recombination coefficient K_r for the carbon-steel back face,
    m^4/(mol s).  Used as the downstream surface kinetics in the analytical
    R = K_d_back / K_d_front ratio."""
    p = MATERIALS["carbon_steel"]
    T = np.asarray(T, dtype=float)
    return p["Kr0"] * np.exp(-p["E_Kr"] / (K_B_EV * T))


def Kd_from_Kr(Kr_mol, Ks_mol):
    """K_d = K_r * K_s^2  (mol/m^2/s/Pa).  All inputs in mol units."""
    return Kr_mol * Ks_mol * Ks_mol


# --- Anderl bracket for the W K_r sweep -------------------------------------

# Anderl 1992 clean polycrystalline W:  K_r_0 = 3.2e-15 m^4/atom/s, E = 1.16 eV
# Anderl 1999 native-oxide W:           K_r_0 ~ 1e-17 m^4/atom/s, E ~ 1.50 eV
# step1_kr_fit.py walks K_r between these two endpoints.

ANDERL_BRACKET = {
    "clean": {
        "K_r_0": 3.2e-15 * N_A,    # mol units
        "E_K_r": 1.16,
        "source": "Anderl 1992 JNM 196-198 p986",
    },
    "native_oxide": {
        "K_r_0": 1.0e-17 * N_A,
        "E_K_r": 1.50,
        "source": "Anderl 1999 JNM 273 p1; Ogorodnikova 2003",
    },
}


def Kr_W_anderl_clean(T):
    p = ANDERL_BRACKET["clean"]
    return p["K_r_0"] * np.exp(-p["E_K_r"] / (K_B_EV * np.asarray(T, dtype=float)))


def Kr_W_anderl_oxide(T):
    p = ANDERL_BRACKET["native_oxide"]
    return p["K_r_0"] * np.exp(-p["E_K_r"] / (K_B_EV * np.asarray(T, dtype=float)))


if __name__ == "__main__":
    print("                            T=547.7K        T=597.2K")
    print(f"  Phi_W (mol/m/s/Pa^0.5):  {Phi('tungsten', 547.7):.3e}    "
          f"{Phi('tungsten', 597.2):.3e}")
    print(f"  Phi_steel:               {Phi('carbon_steel', 547.7):.3e}    "
          f"{Phi('carbon_steel', 597.2):.3e}")
    print(f"  K_r_W Anderl clean:      {Kr_W_anderl_clean(547.7):.3e}    "
          f"{Kr_W_anderl_clean(597.2):.3e}")
    print(f"  K_r_W Anderl oxide:      {Kr_W_anderl_oxide(547.7):.3e}    "
          f"{Kr_W_anderl_oxide(597.2):.3e}")

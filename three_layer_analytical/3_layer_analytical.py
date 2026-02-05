"""
Three-Layer Permeation Analysis
Based on the derivation extending the two-layer Transport Models

===============================================================================
SYSTEM CONFIGURATION
===============================================================================

    Gas (p_1) | Material 1 (e_1, D_1, K_s1) | Material 2 (e_2, D_2, K_s2) | Material 3 (e_3, D_3, K_s3) | Vacuum

    Where:
    - p_1     : upstream gas pressure (Pa)
    - e_1, e_2, e_3: layer thicknesses (m)
    - D_1, D_2, D_3: diffusion coefficients (m²/s)
    - K_s1, K_s2, K_s3: Sieverts' constants (mol/m³/Pa^0.5)
    - K_d1, K_d3: dissociation constants at surfaces (mol/m²/s/Pa)
    - K_r1, K_r3: recombination constants at surfaces (m⁴/mol/s)

===============================================================================
DIMENSIONAL EQUATIONS
===============================================================================

    Flux Balance at Steady State:
        K_d1·p_1 - K_r1·C_1² = J_1 = J_2 = J_3 = K_r3·C_3²

    Where:
    - Left term:  dissociation flux at gas-material1 surface
    - J_1, J_2, J_3: diffusive flux through materials 1, 2, and 3
    - Right term: recombination flux at material3-vacuum surface

===============================================================================
NON-DIMENSIONALIZATION
===============================================================================

    Dimensionless Concentrations:
        u = C_3 / (K_s3·√p_1)    (concentration at vacuum side, 0 ≤ u ≤ 1)
        v = C_1 / (K_s1·√p_1)    (concentration at gas side, 0 ≤ v ≤ 1)

    Dimensionless Parameters:
        W = (W1 + W2)(W2 + W3)
            → For three layers, W is a product of two sums
            → Note: W2 (middle layer) appears in BOTH factors

        Individual layer contributions (same form as two-layer):
        W1 = K_d1·√p_1·e_1 / (D_1·K_s1)
            → Diffusion resistance contribution from layer 1

        W2 = K_d1·√p_1·e_2 / (D_2·K_s2)
            → Diffusion resistance contribution from layer 2 (middle)
            → Appears in both (W1+W2) and (W2+W3) terms

        W3 = K_d1·√p_1·e_3 / (D_3·K_s3)
            → Diffusion resistance contribution from layer 3

        R = K_d3 / K_d1
            → Ratio of dissociation constants (outermost to innermost)
            → Determines relative importance of the two surfaces

===============================================================================
DIMENSIONLESS FLUX BALANCE EQUATION
===============================================================================

    After non-dimensionalization, the flux balance becomes:

        W·(1 - v²) = v - u = W·R·u²

    This has the SAME structure as the two-layer case, but with:
        W = (W1 + W2)(W2 + W3)

    The dimensionless flux is:
        J* = J / J_DL = v - u

    So J* = 1 means the actual flux equals the diffusion-limited flux.

===============================================================================
LIMITING REGIMES
===============================================================================

    DIFFUSION-LIMITED (W >> 1):
        - Surface kinetics is fast, diffusion is the bottleneck
        - Concentrations: v → 1, u → 0
        - Flux: J* = v - u → 1
        - Physical meaning: J = J_DL

    SURFACE-LIMITED (W << 1):
        - Diffusion is fast, surface kinetics is the bottleneck
        - Concentrations: v ≈ u (small gradient)
        - From the equations: v ≈ u ≈ 1/√(1+R)
        - Flux: J* = W·R/(1+R)
        - Physical meaning: J << J_DL

===============================================================================
SPECIAL ROLE OF MIDDLE LAYER (W2)
===============================================================================

    Since W = (W1 + W2)(W2 + W3), the middle layer W2 appears twice.

    This means:
    - Increasing W2 affects BOTH factors, potentially having a larger effect
    - The middle layer can act as a "bridge" or "bottleneck"
    - If W2 is very small: W ≈ W1·W3 (outer layers dominate)
    - If W2 is very large: W ≈ W2² (middle layer dominates both terms)

===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Create figs directory if it doesn't exist
FIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)


# =============================================================================
# DIMENSIONLESS PARAMETER COMPUTATION (W1, W2, W3)
# =============================================================================

def compute_W1(K_d1, p_1, e_1, D_1, K_s1):
    """
    Compute W1: the diffusion resistance contribution from layer 1.

    W1 = K_d1·√p_1·e_1 / (D_1·K_s1)

    Physical meaning:
        - Ratio of upstream surface kinetics rate to layer 1 diffusion rate
        - Large W1 means layer 1 is a significant diffusion bottleneck
        - Small W1 means layer 1 offers little resistance to diffusion

    Parameters
    ----------
    K_d1 : float
        Dissociation constant at gas-material1 surface (mol/m²/s/Pa)
    p_1 : float
        Upstream gas pressure (Pa)
    e_1 : float
        Thickness of layer 1 (m)
    D_1 : float
        Diffusion coefficient in layer 1 (m²/s)
    K_s1 : float
        Sieverts' constant for layer 1 (mol/m³/Pa^0.5)

    Returns
    -------
    W1 : float
        Dimensionless diffusion resistance for layer 1
    """
    return K_d1 * np.sqrt(p_1) * e_1 / (D_1 * K_s1)


def compute_W2(K_d1, p_1, e_2, D_2, K_s2):
    """
    Compute W2: the diffusion resistance contribution from layer 2 (middle).

    W2 = K_d1·√p_1·e_2 / (D_2·K_s2)

    Physical meaning:
        - Ratio of upstream surface kinetics rate to layer 2 diffusion rate
        - W2 appears in BOTH factors of W = (W1+W2)(W2+W3)
        - This gives the middle layer a special role in the three-layer system

    Parameters
    ----------
    K_d1 : float
        Dissociation constant at gas-material1 surface (mol/m²/s/Pa)
    p_1 : float
        Upstream gas pressure (Pa)
    e_2 : float
        Thickness of layer 2 (m)
    D_2 : float
        Diffusion coefficient in layer 2 (m²/s)
    K_s2 : float
        Sieverts' constant for layer 2 (mol/m³/Pa^0.5)

    Returns
    -------
    W2 : float
        Dimensionless diffusion resistance for layer 2
    """
    return K_d1 * np.sqrt(p_1) * e_2 / (D_2 * K_s2)


def compute_W3(K_d1, p_1, e_3, D_3, K_s3):
    """
    Compute W3: the diffusion resistance contribution from layer 3.

    W3 = K_d1·√p_1·e_3 / (D_3·K_s3)

    Physical meaning:
        - Ratio of upstream surface kinetics rate to layer 3 diffusion rate
        - Large W3 means layer 3 is a significant diffusion bottleneck
        - Small W3 means layer 3 offers little resistance to diffusion

    Note: W3 uses K_d1 (not K_d3) because W is normalized by the upstream
    surface kinetics. This allows consistent combination with W1 and W2.

    Parameters
    ----------
    K_d1 : float
        Dissociation constant at gas-material1 surface (mol/m²/s/Pa)
    p_1 : float
        Upstream gas pressure (Pa)
    e_3 : float
        Thickness of layer 3 (m)
    D_3 : float
        Diffusion coefficient in layer 3 (m²/s)
    K_s3 : float
        Sieverts' constant for layer 3 (mol/m³/Pa^0.5)

    Returns
    -------
    W3 : float
        Dimensionless diffusion resistance for layer 3
    """
    return K_d1 * np.sqrt(p_1) * e_3 / (D_3 * K_s3)


def compute_W_total(W1, W2, W3):
    """
    Compute total W from individual layer contributions.

    W = (W1 + W2)(W2 + W3)

    This is the key difference from two-layer: W is a product, not a sum.

    Parameters
    ----------
    W1 : float
        Layer 1 diffusion resistance
    W2 : float
        Layer 2 diffusion resistance (middle layer)
    W3 : float
        Layer 3 diffusion resistance

    Returns
    -------
    W : float
        Total permeation parameter
    """
    return (W1 + W2) * (W2 + W3)


def compute_W_from_physical(K_d1, p_1, e_1, e_2, e_3, D_1, D_2, D_3, K_s1, K_s2, K_s3):
    """
    Compute total W and its components W1, W2, W3 from physical parameters.

    Returns W, W1, W2, W3 where W = (W1 + W2)(W2 + W3).

    Parameters
    ----------
    K_d1 : float
        Dissociation constant at gas-material1 surface (mol/m²/s/Pa)
    p_1 : float
        Upstream gas pressure (Pa)
    e_1, e_2, e_3 : float
        Layer thicknesses (m)
    D_1, D_2, D_3 : float
        Diffusion coefficients (m²/s)
    K_s1, K_s2, K_s3 : float
        Sieverts' constants (mol/m³/Pa^0.5)

    Returns
    -------
    W : float
        Total permeation parameter = (W1 + W2)(W2 + W3)
    W1 : float
        Layer 1 contribution
    W2 : float
        Layer 2 contribution (middle)
    W3 : float
        Layer 3 contribution
    """
    W1 = compute_W1(K_d1, p_1, e_1, D_1, K_s1)
    W2 = compute_W2(K_d1, p_1, e_2, D_2, K_s2)
    W3 = compute_W3(K_d1, p_1, e_3, D_3, K_s3)
    W = compute_W_total(W1, W2, W3)
    return W, W1, W2, W3


# =============================================================================
# CORE PHYSICS: Dimensionless Equations and Solver
# =============================================================================

def dimensionless_equations(x, W, R):
    """
    System of dimensionless equations to be solved.

    The dimensionless flux balance is:
        W·(1 - v²) = v - u = W·R·u²

    This represents THREE equal quantities:
        - W·(1 - v²): net flux at upstream surface (dissociation - recombination)
        - v - u: diffusive flux through the three-layer system
        - W·R·u²: recombination flux at downstream surface

    We split this into two equations:
        eq1: W·(1 - v²) - (v - u) = 0
        eq2: (v - u) - W·R·u² = 0

    Parameters
    ----------
    x : array [u, v]
        u: dimensionless concentration at vacuum side (C_3 / K_s3·√p_1)
        v: dimensionless concentration at gas side (C_1 / K_s1·√p_1)
    W : float
        Permeation parameter = (W1+W2)(W2+W3)
        W >> 1 means diffusion-limited, W << 1 means surface-limited
    R : float
        Ratio of dissociation constants: R = K_d3 / K_d1

    Returns
    -------
    residuals : array [eq1, eq2]
        Both should be zero at the solution
    """
    u, v = x

    # Equation (a): upstream surface flux = diffusive flux
    eq1 = W * (1 - v**2) - (v - u)

    # Equation (b): diffusive flux = downstream surface flux
    eq2 = (v - u) - W * R * u**2

    return [eq1, eq2]


def solve_three_layer(W, R):
    """
    Solve for dimensionless concentrations u and v given W and R.

    This is the "full numerical solution" - we find the exact values of u and v
    that satisfy both equations simultaneously, without making any approximations.

    The system W(1-v²) = v - u = WRu² is reduced to a quartic in u:
        W²R²u⁴ + 2WRu³ + (1+R)u² - 1 = 0

    Once u is found, v and J* are computed as:
        v = u + WRu²
        J* = v - u = WRu²

    Parameters
    ----------
    W : float
        Permeation parameter = (W1+W2)(W2+W3)
    R : float
        Dissociation constant ratio = K_d3/K_d1

    Returns
    -------
    u : float
        Dimensionless concentration at vacuum side (0 ≤ u ≤ 1)
    v : float
        Dimensionless concentration at gas side (0 ≤ v ≤ 1)
    J_star : float
        Dimensionless flux = v - u
        J* = 1 means flux equals diffusion-limited flux
        J* < 1 means flux is below diffusion-limited value
    """
    # Solve the quartic equation: W²R²u⁴ + 2WRu³ + (1+R)u² - 1 = 0
    # Coefficients for np.roots: [a4, a3, a2, a1, a0] for a4*x^4 + a3*x^3 + ...
    a4 = W**2 * R**2
    a3 = 2 * W * R
    a2 = 1 + R
    a1 = 0
    a0 = -1

    try:
        # Find all roots of the quartic
        roots = np.roots([a4, a3, a2, a1, a0])

        # Find the physical root (real and positive, with u <= 1)
        physical_u = None
        for root in roots:
            # Check if root is real (negligible imaginary part)
            if np.abs(root.imag) < 1e-10:
                real_root = root.real
                # Check if root is positive and physical (0 < u <= 1)
                if real_root > 0 and real_root <= 1 + 1e-10:
                    physical_u = real_root
                    break

        if physical_u is None:
            return np.nan, np.nan, np.nan

        u = min(physical_u, 1.0)  # Ensure u <= 1

        # Compute v and J* from u
        v = u + W * R * u**2
        J_star = v - u  # = W * R * u²

        # Ensure physical bounds
        v = max(0, min(1, v))

        return u, v, J_star
    except:
        return np.nan, np.nan, np.nan


def solve_three_layer_W1W2W3(W1, W2, W3, R):
    """
    Solve for dimensionless concentrations using W1, W2, W3 separately.

    This is a convenience wrapper that accepts the decomposed W parameters,
    allowing analysis of how each layer contributes to the overall behavior.

    Parameters
    ----------
    W1 : float
        Layer 1 diffusion resistance
    W2 : float
        Layer 2 diffusion resistance (middle layer)
    W3 : float
        Layer 3 diffusion resistance
    R : float
        Dissociation constant ratio: K_d3 / K_d1

    Returns
    -------
    u : float
        Dimensionless concentration at vacuum side
    v : float
        Dimensionless concentration at gas side
    J_star : float
        Dimensionless flux = v - u
    layer_info : dict
        Information about layer dominance and contributions
    """
    W = compute_W_total(W1, W2, W3)
    u, v, J_star = solve_three_layer(W, R)

    # Analyze layer contributions
    term1 = W1 + W2  # First factor
    term2 = W2 + W3  # Second factor

    # Determine which factor dominates
    if term1 > 0 and term2 > 0:
        ratio = term1 / term2
        if ratio > 10:
            term_dominance = 'term1 (W1+W2)'
        elif ratio < 0.1:
            term_dominance = 'term2 (W2+W3)'
        else:
            term_dominance = 'comparable'
    else:
        term_dominance = 'undefined'

    # Determine middle layer significance
    # W2 appears in both terms, so check its relative contribution
    if W1 + W2 + W3 > 0:
        w2_fraction = W2 / (W1 + W2 + W3)
        if w2_fraction > 0.5:
            middle_layer_role = 'dominant'
        elif w2_fraction > 0.2:
            middle_layer_role = 'significant'
        else:
            middle_layer_role = 'minor'
    else:
        middle_layer_role = 'undefined'

    layer_info = {
        'W': W,
        'W1': W1,
        'W2': W2,
        'W3': W3,
        'term1': term1,
        'term2': term2,
        'term_dominance': term_dominance,
        'middle_layer_role': middle_layer_role
    }

    return u, v, J_star, layer_info


# =============================================================================
# LIMITING REGIME APPROXIMATIONS
# =============================================================================

def flux_DL(W, R):
    """
    Diffusion-Limited (DL) flux approximation.

    When W >> 1, surface kinetics is very fast compared to diffusion.
    The surface reactions reach equilibrium instantly, so:
        - At gas side: C_1 = K_s1·√p_1, giving v = 1
        - At vacuum side: C_3 = 0, giving u = 0

    Therefore: J* = v - u = 1 - 0 = 1

    This approximation is valid when W > ~100 (error < 5%)
    """
    return 1.0


def flux_SL(W, R):
    """
    Surface-Limited (SL) flux approximation.

    When W << 1, diffusion is very fast compared to surface kinetics.
    The concentration gradient becomes small (v ≈ u), and surface
    processes become the bottleneck.

    Derivation:
        From v - u = W·R·u² and W small: v ≈ u (gradient vanishes)
        From W·(1 - v²) = v - u and v ≈ u:
            W·(1 - v²) ≈ W·R·v²  (since u ≈ v)
            1 - v² ≈ R·v²
            1 ≈ v²·(1 + R)
            v ≈ 1/√(1 + R)

        Therefore:
            J* = W·(1 - v²) = W·(1 - 1/(1+R)) = W·R/(1+R)

    This approximation is valid when W < ~0.01 (error < 5%)

    Physical meaning: flux is proportional to W (and thus to surface kinetics)
    """
    J_star_sl = W * R / (1 + R)
    return J_star_sl


# =============================================================================
# ERROR CALCULATION
# =============================================================================

def relative_error(J_approx, J_full):
    """
    Calculate the relative error between an approximation and the full solution.

    err = |J*_approx - J*_full| / J*_full

    We use err < 0.05 (5%) as the threshold for when an approximation
    is considered acceptable.

    Parameters
    ----------
    J_approx : float
        Flux from limiting regime approximation (J*_DL or J*_SL)
    J_full : float
        Flux from full numerical solution

    Returns
    -------
    err : float
        Relative error (0 = perfect match, 1 = 100% error)
    """
    if J_full == 0 or np.isnan(J_full):
        return np.inf
    return np.abs(J_approx - J_full) / np.abs(J_full)


def compute_regime_map(W_range, R_range):
    """
    Compute error maps for DL and SL approximations across parameter space.

    This function sweeps through all combinations of W and R, and for each point:
        1. Solves the full system numerically to get J*_full
        2. Calculates the DL approximation: J*_DL = 1
        3. Calculates the SL approximation: J*_SL = W·R/(1+R)
        4. Computes relative errors for both approximations

    Parameters
    ----------
    W_range : array
        Array of W values to evaluate (typically logspace from 10^-5 to 10^5)
    R_range : array
        Array of R values to evaluate (typically logspace from 10^-3 to 10^3)

    Returns
    -------
    W_grid, R_grid : 2D arrays
        Meshgrid of W and R values
    error_DL : 2D array
        Relative error of DL approximation at each (W, R) point
    error_SL : 2D array
        Relative error of SL approximation at each (W, R) point
    J_full_grid : 2D array
        Full numerical solution at each (W, R) point
    """
    W_grid, R_grid = np.meshgrid(W_range, R_range)

    error_DL = np.zeros_like(W_grid)
    error_SL = np.zeros_like(W_grid)
    J_full_grid = np.zeros_like(W_grid)

    for i in range(len(R_range)):
        for j in range(len(W_range)):
            W = W_range[j]
            R = R_range[i]

            u, v, J_full = solve_three_layer(W, R)
            J_full_grid[i, j] = J_full

            if not np.isnan(J_full) and J_full > 0:
                J_DL = flux_DL(W, R)
                J_SL = flux_SL(W, R)

                error_DL[i, j] = relative_error(J_DL, J_full)
                error_SL[i, j] = relative_error(J_SL, J_full)
            else:
                error_DL[i, j] = np.nan
                error_SL[i, j] = np.nan

    return W_grid, R_grid, error_DL, error_SL, J_full_grid


# =============================================================================
# PLOTTING FUNCTIONS - BASIC (similar to 2-layer)
# =============================================================================

def plot_regime_map(W_range=None, R_range=None):
    """
    Plot regime map: W vs R showing where DL and SL approximations are valid.

    This creates three panels:
        1. DL error map: shows where J* = 1 is a good approximation
        2. SL error map: shows where J* = W·R/(1+R) is a good approximation
        3. Best regime: shows which approximation has lower error at each point

    The red contour lines mark the 5% error boundary.
    """
    if W_range is None:
        W_range = np.logspace(-6, 6, 100)
    if R_range is None:
        R_range = np.logspace(-6, 6, 100)

    W_grid, R_grid, error_DL, error_SL, J_full = compute_regime_map(W_range, R_range)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: DL approximation error
    ax1 = axes[0]
    err_DL_clipped = np.clip(error_DL, 1e-6, 0.5)
    pcm1 = ax1.pcolormesh(W_grid, R_grid, err_DL_clipped,
                          norm=plt.matplotlib.colors.LogNorm(vmin=1e-4, vmax=0.5),
                          cmap='viridis_r', shading='auto')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('$W = (W_1+W_2)(W_2+W_3)$ (-)')
    ax1.set_ylabel('R = $K_{d3}/K_{d1}$ (-)')
    ax1.set_title('Diffusion-Limited Error\n$J^* = 1$')
    plt.colorbar(pcm1, ax=ax1, label='Relative Error')
    ax1.contour(W_grid, R_grid, error_DL, levels=[0.05], colors='red', linewidths=2)

    # Panel 2: SL approximation error
    ax2 = axes[1]
    err_SL_clipped = np.clip(error_SL, 1e-6, 0.5)
    pcm2 = ax2.pcolormesh(W_grid, R_grid, err_SL_clipped,
                          norm=plt.matplotlib.colors.LogNorm(vmin=1e-4, vmax=0.5),
                          cmap='viridis_r', shading='auto')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('$W = (W_1+W_2)(W_2+W_3)$ (-)')
    ax2.set_ylabel('R = $K_{d3}/K_{d1}$ (-)')
    ax2.set_title('Surface-Limited Error\n$J^* = W·R/(1+R)$')
    plt.colorbar(pcm2, ax=ax2, label='Relative Error')
    ax2.contour(W_grid, R_grid, error_SL, levels=[0.05], colors='red', linewidths=2)

    # Panel 3: Best regime map
    ax3 = axes[2]
    min_error = np.minimum(error_DL, error_SL)
    best_regime = np.where(error_DL < error_SL, 0, 1)
    mixed_regime_mask = min_error >= 0.05
    best_regime_masked = np.ma.masked_where(mixed_regime_mask, best_regime)

    cmap = plt.colormaps.get_cmap('RdYlBu').resampled(2)
    cmap.set_bad(color='white')
    pcm3 = ax3.pcolormesh(W_grid, R_grid, best_regime_masked, cmap=cmap, shading='auto', vmin=-0.5, vmax=1.5)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('$W = (W_1+W_2)(W_2+W_3)$ (-)')
    ax3.set_ylabel('R = $K_{d3}/K_{d1}$ (-)')
    ax3.set_title('Best Limiting Regime\n(white = mixed regime)')
    cbar = plt.colorbar(pcm3, ax=ax3, ticks=[0, 1])
    cbar.ax.set_yticklabels(['DL', 'SL'])
    ax3.contour(W_grid, R_grid, min_error, levels=[0.05], colors='black', linewidths=2, linestyles='--')

    plt.tight_layout()
    return fig


def plot_flux_vs_W(R_values=[0.01, 0.1, 1, 10, 100]):
    """
    Plot dimensionless flux J* vs W for different R values.

    This shows how the flux transitions from surface-limited (J* ∝ W)
    to diffusion-limited (J* → 1) as W increases.
    """
    W_range = np.logspace(-4, 4, 100)

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.viridis(np.linspace(0, 1, len(R_values)))

    for R, color in zip(R_values, colors):
        J_full = []

        for W in W_range:
            _, _, J = solve_three_layer(W, R)
            J_full.append(J)

        ax.loglog(W_range, J_full, '-', color=color, linewidth=2, label=f'R = {R}')

    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='DL: $J^*=1$')
    ax.loglog(W_range, W_range * 0.5, 'k:', alpha=0.5, label='SL: $J^* ∝ W$')

    ax.set_xlabel('$W = (W_1+W_2)(W_2+W_3)$ (-)', fontsize=12)
    ax.set_ylabel('$J^*$ (-)', fontsize=12)
    ax.set_title('Dimensionless Flux vs Permeation Parameter (3-Layer)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1e-4, 1e4])
    ax.set_ylim([1e-5, 10])

    return fig


def plot_error_1D(R=1.0):
    """
    Plot error vs W for a fixed R.

    This clearly shows:
        - For W < 0.01: SL approximation valid (red line below 5%)
        - For W > 100: DL approximation valid (blue line below 5%)
        - For 0.01 < W < 100: Mixed regime (both approximations fail)
    """
    W_range = np.logspace(-5, 5, 200)

    err_DL = []
    err_SL = []

    for W in W_range:
        _, _, J_full = solve_three_layer(W, R)

        if not np.isnan(J_full) and J_full > 0:
            err_DL.append(relative_error(flux_DL(W, R), J_full))
            err_SL.append(relative_error(flux_SL(W, R), J_full))
        else:
            err_DL.append(np.nan)
            err_SL.append(np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))

    err_DL = np.array(err_DL)
    err_SL = np.array(err_SL)

    ax.fill_between(W_range, 1e-6, 10, where=(err_SL < 0.05),
                    alpha=0.2, color='red', label='SL regime (<5% error)')
    ax.fill_between(W_range, 1e-6, 10, where=(err_DL < 0.05),
                    alpha=0.2, color='blue', label='DL regime (<5% error)')

    ax.loglog(W_range, err_DL, 'b-', linewidth=2, label='DL approximation ($J^*=1$)')
    ax.loglog(W_range, err_SL, 'r-', linewidth=2, label='SL approximation ($J^*=WR/(1+R)$)')
    ax.axhline(y=0.05, color='gray', linestyle='--', label='5% error threshold')

    ax.set_xlabel('$W = (W_1+W_2)(W_2+W_3)$ (-)', fontsize=12)
    ax.set_ylabel('Relative Error (-)', fontsize=12)
    ax.set_title(f'Relative Error vs W (3-Layer, R = {R})', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1e-5, 1e5])
    ax.set_ylim([1e-6, 10])

    return fig


# =============================================================================
# PLOTTING FUNCTIONS - 3-LAYER SPECIFIC
# =============================================================================

def plot_regime_map_W1W3(W1_range=None, W3_range=None, W2_values=[0.01, 0.1, 1, 10], R=1.0):
    """
    Plot regime maps in W1-W3 space for different fixed W2 values.

    This shows how the outer layers (W1 and W3) interact for different
    middle layer contributions (W2).

    Since W = (W1+W2)(W2+W3), the middle layer affects both terms.

    Parameters
    ----------
    W1_range : array, optional
        W1 values to evaluate. Default: logspace(-3, 3)
    W3_range : array, optional
        W3 values to evaluate. Default: logspace(-3, 3)
    W2_values : list
        Fixed W2 values for each panel
    R : float
        Dissociation constant ratio
    """
    if W1_range is None:
        W1_range = np.logspace(-3, 3, 80)
    if W3_range is None:
        W3_range = np.logspace(-3, 3, 80)

    n_panels = len(W2_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(4*n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    for idx, W2 in enumerate(W2_values):
        ax = axes[idx]

        W1_grid, W3_grid = np.meshgrid(W1_range, W3_range)
        J_grid = np.zeros_like(W1_grid)
        error_DL_grid = np.zeros_like(W1_grid)
        error_SL_grid = np.zeros_like(W1_grid)

        for i in range(len(W3_range)):
            for j in range(len(W1_range)):
                W1 = W1_range[j]
                W3 = W3_range[i]
                W = compute_W_total(W1, W2, W3)

                _, _, J_full = solve_three_layer(W, R)
                J_grid[i, j] = J_full

                if not np.isnan(J_full) and J_full > 0:
                    error_DL_grid[i, j] = relative_error(flux_DL(W, R), J_full)
                    error_SL_grid[i, j] = relative_error(flux_SL(W, R), J_full)
                else:
                    error_DL_grid[i, j] = np.nan
                    error_SL_grid[i, j] = np.nan

        # Plot J* contours
        J_clipped = np.clip(J_grid, 1e-6, 1.0)
        pcm = ax.pcolormesh(W1_grid, W3_grid, J_clipped,
                            norm=plt.matplotlib.colors.LogNorm(vmin=1e-4, vmax=1),
                            cmap='plasma', shading='auto')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('$W_1$ (Layer 1)')
        ax.set_ylabel('$W_3$ (Layer 3)')
        ax.set_title(f'$W_2$ = {W2}')

        # Add regime boundaries
        min_error = np.minimum(error_DL_grid, error_SL_grid)
        ax.contour(W1_grid, W3_grid, min_error, levels=[0.05],
                   colors='white', linewidths=2, linestyles='--')

        # Add W1=W3 line
        ax.plot([1e-3, 1e3], [1e-3, 1e3], 'w:', alpha=0.5)

        plt.colorbar(pcm, ax=ax, label='$J^*$')

    plt.suptitle(f'Dimensionless Flux in $W_1$-$W_3$ Space (R = {R})\n'
                 f'Dashed line: 5% error boundary', fontsize=12)
    plt.tight_layout()
    return fig


def plot_middle_layer_sensitivity(W1_values=[0.1, 1, 10], W3_values=[0.1, 1, 10], R=1.0):
    """
    Analyze how the middle layer (W2) affects the flux.

    Since W = (W1+W2)(W2+W3), W2 appears in both factors.
    This plot shows how J* varies with W2 for different (W1, W3) combinations.

    Parameters
    ----------
    W1_values : list
        Fixed W1 values to explore
    W3_values : list
        Fixed W3 values to explore
    R : float
        Dissociation constant ratio
    """
    W2_range = np.logspace(-4, 4, 100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: J* vs W2 for different (W1, W3) combinations
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(W1_values) * len(W3_values)))
    color_idx = 0

    for W1 in W1_values:
        for W3 in W3_values:
            J_values = []
            for W2 in W2_range:
                W = compute_W_total(W1, W2, W3)
                _, _, J = solve_three_layer(W, R)
                J_values.append(J)

            ax1.loglog(W2_range, J_values, '-', color=colors[color_idx], linewidth=2,
                       label=f'$W_1$={W1}, $W_3$={W3}')
            color_idx += 1

    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='DL limit')
    ax1.set_xlabel('$W_2$ (Middle Layer)', fontsize=12)
    ax1.set_ylabel('$J^*$', fontsize=12)
    ax1.set_title(f'Flux vs Middle Layer Resistance (R = {R})', fontsize=14)
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1e-4, 1e4])
    ax1.set_ylim([1e-5, 2])

    # Panel 2: Total W vs W2 showing the quadratic nature
    ax2 = axes[1]
    color_idx = 0
    for W1 in W1_values:
        for W3 in W3_values:
            W_values = [compute_W_total(W1, W2, W3) for W2 in W2_range]
            ax2.loglog(W2_range, W_values, '-', color=colors[color_idx], linewidth=2,
                       label=f'$W_1$={W1}, $W_3$={W3}')
            color_idx += 1

    # Reference lines
    ax2.loglog(W2_range, W2_range**2, 'k--', alpha=0.5, label='$W_2^2$ (W2 >> W1, W3)')

    ax2.set_xlabel('$W_2$ (Middle Layer)', fontsize=12)
    ax2.set_ylabel('$W = (W_1+W_2)(W_2+W_3)$', fontsize=12)
    ax2.set_title('Total W vs Middle Layer', fontsize=14)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([1e-4, 1e4])

    plt.tight_layout()
    return fig


def plot_flux_vs_individual_Wi(R=1.0):
    """
    Plot J* vs each individual Wi while holding the others fixed.

    Three panels showing:
    1. J* vs W1 for fixed (W2, W3)
    2. J* vs W2 for fixed (W1, W3) - most interesting due to W2's special role
    3. J* vs W3 for fixed (W1, W2)

    Parameters
    ----------
    R : float
        Dissociation constant ratio
    """
    W_range = np.logspace(-4, 4, 100)
    fixed_values = [0.01, 0.1, 1, 10]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(fixed_values)))

    # Panel 1: J* vs W1 (fix W2=1, vary W3)
    ax1 = axes[0]
    W2_fixed = 1.0
    for W3_fixed, color in zip(fixed_values, colors):
        J_values = []
        for W1 in W_range:
            W = compute_W_total(W1, W2_fixed, W3_fixed)
            _, _, J = solve_three_layer(W, R)
            J_values.append(J)
        ax1.loglog(W_range, J_values, '-', color=color, linewidth=2, label=f'$W_3$={W3_fixed}')

    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax1.set_xlabel('$W_1$ (Layer 1)', fontsize=12)
    ax1.set_ylabel('$J^*$', fontsize=12)
    ax1.set_title(f'$J^*$ vs $W_1$ ($W_2$={W2_fixed} fixed)', fontsize=12)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1e-4, 1e4])
    ax1.set_ylim([1e-5, 2])

    # Panel 2: J* vs W2 (fix W1=1, vary W3) - MOST INTERESTING
    ax2 = axes[1]
    W1_fixed = 1.0
    for W3_fixed, color in zip(fixed_values, colors):
        J_values = []
        for W2 in W_range:
            W = compute_W_total(W1_fixed, W2, W3_fixed)
            _, _, J = solve_three_layer(W, R)
            J_values.append(J)
        ax2.loglog(W_range, J_values, '-', color=color, linewidth=2, label=f'$W_3$={W3_fixed}')

    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax2.set_xlabel('$W_2$ (Middle Layer)', fontsize=12)
    ax2.set_ylabel('$J^*$', fontsize=12)
    ax2.set_title(f'$J^*$ vs $W_2$ ($W_1$={W1_fixed} fixed)\n'
                  '(W2 appears in both terms!)', fontsize=12)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([1e-4, 1e4])
    ax2.set_ylim([1e-5, 2])

    # Panel 3: J* vs W3 (fix W2=1, vary W1)
    ax3 = axes[2]
    W2_fixed = 1.0
    for W1_fixed, color in zip(fixed_values, colors):
        J_values = []
        for W3 in W_range:
            W = compute_W_total(W1_fixed, W2_fixed, W3)
            _, _, J = solve_three_layer(W, R)
            J_values.append(J)
        ax3.loglog(W_range, J_values, '-', color=color, linewidth=2, label=f'$W_1$={W1_fixed}')

    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax3.set_xlabel('$W_3$ (Layer 3)', fontsize=12)
    ax3.set_ylabel('$J^*$', fontsize=12)
    ax3.set_title(f'$J^*$ vs $W_3$ ($W_2$={W2_fixed} fixed)', fontsize=12)
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([1e-4, 1e4])
    ax3.set_ylim([1e-5, 2])

    plt.suptitle(f'Flux Dependence on Individual Layer Resistances (R = {R})', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_iso_W_contours(R=1.0):
    """
    Plot iso-W contours in W1-W3 space for fixed W2.

    Shows lines of constant W = (W1+W2)(W2+W3), which are hyperbolas
    in W1-W3 space (unlike the straight lines W1+W2=const in 2-layer).

    Parameters
    ----------
    R : float
        Dissociation constant ratio
    """
    W1_range = np.logspace(-3, 3, 100)
    W3_range = np.logspace(-3, 3, 100)
    W2_values = [0.1, 1.0, 10.0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, W2 in enumerate(W2_values):
        ax = axes[idx]
        W1_grid, W3_grid = np.meshgrid(W1_range, W3_range)

        # Compute J* for each point
        J_grid = np.zeros_like(W1_grid)
        W_grid = np.zeros_like(W1_grid)

        for i in range(len(W3_range)):
            for j in range(len(W1_range)):
                W1 = W1_range[j]
                W3 = W3_range[i]
                W = compute_W_total(W1, W2, W3)
                W_grid[i, j] = W

                _, _, J = solve_three_layer(W, R)
                J_grid[i, j] = J

        # Plot J* as color
        J_clipped = np.clip(J_grid, 1e-4, 1.0)
        pcm = ax.pcolormesh(W1_grid, W3_grid, J_clipped,
                            norm=plt.matplotlib.colors.LogNorm(vmin=1e-4, vmax=1),
                            cmap='plasma', shading='auto')
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Add iso-W contours
        W_levels = [0.01, 0.1, 1, 10, 100, 1000]
        contours = ax.contour(W1_grid, W3_grid, W_grid, levels=W_levels,
                              colors='white', linewidths=1.5, linestyles='-')
        ax.clabel(contours, inline=True, fontsize=8, fmt='W=%.2g')

        # Add W1=W3 line for reference
        ax.plot([1e-3, 1e3], [1e-3, 1e3], 'w--', alpha=0.5, label='$W_1=W_3$')

        ax.set_xlabel('$W_1$ (Layer 1)', fontsize=12)
        ax.set_ylabel('$W_3$ (Layer 3)', fontsize=12)
        ax.set_title(f'$W_2$ = {W2}', fontsize=12)
        plt.colorbar(pcm, ax=ax, label='$J^*$')

    plt.suptitle(f'Iso-W Contours in $W_1$-$W_3$ Space (R = {R})\n'
                 f'$W = (W_1+W_2)(W_2+W_3)$', fontsize=14)
    plt.tight_layout()
    return fig


def plot_layer_dominance_analysis(R=1.0):
    """
    Analyze which term - (W1+W2) or (W2+W3) - dominates the behavior.

    For three-layer: W = (W1+W2)(W2+W3)
    - If (W1+W2) >> (W2+W3): upstream layers dominate
    - If (W2+W3) >> (W1+W2): downstream layers dominate
    - If (W1+W2) ≈ (W2+W3): balanced contribution

    Parameters
    ----------
    R : float
        Dissociation constant ratio
    """
    W1_range = np.logspace(-3, 3, 80)
    W3_range = np.logspace(-3, 3, 80)
    W2_values = [0.01, 0.1, 1.0, 10.0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, W2 in enumerate(W2_values):
        ax = axes[idx]
        W1_grid, W3_grid = np.meshgrid(W1_range, W3_range)

        # Compute term ratio
        term1 = W1_grid + W2  # (W1 + W2)
        term2 = W2 + W3_grid  # (W2 + W3)
        term_ratio = np.log10(term1 / term2)  # log10 for symmetric visualization

        # Plot term ratio
        vmax = 3
        pcm = ax.pcolormesh(W1_grid, W3_grid, term_ratio,
                            cmap='RdBu_r', shading='auto', vmin=-vmax, vmax=vmax)
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Add contours
        ax.contour(W1_grid, W3_grid, term_ratio, levels=[0],
                   colors='black', linewidths=2, linestyles='-')
        ax.contour(W1_grid, W3_grid, term_ratio, levels=[-1, 1],
                   colors='gray', linewidths=1, linestyles='--')

        # Add W1=W3 line
        ax.plot([1e-3, 1e3], [1e-3, 1e3], 'k:', alpha=0.5)

        ax.set_xlabel('$W_1$ (Layer 1)', fontsize=11)
        ax.set_ylabel('$W_3$ (Layer 3)', fontsize=11)
        ax.set_title(f'$W_2$ = {W2}', fontsize=12)

        cbar = plt.colorbar(pcm, ax=ax)
        cbar.set_label('$\\log_{10}[(W_1+W_2)/(W_2+W_3)]$', fontsize=10)

    plt.suptitle(f'Term Dominance Analysis (R = {R})\n'
                 f'Red: upstream dominant, Blue: downstream dominant, '
                 f'Black line: balanced', fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def plot_comparison_2layer_vs_3layer():
    """
    Compare 2-layer (W = W1+W2) vs 3-layer (W = (W1+W2)(W2+W3)) behavior.

    Shows how the flux depends on W for both formulations,
    illustrating the key difference in how layers combine.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    R = 1.0

    # Panel 1: J* vs total W
    ax1 = axes[0]
    W_range = np.logspace(-4, 4, 100)

    J_values = []
    for W in W_range:
        _, _, J = solve_three_layer(W, R)
        J_values.append(J)

    ax1.loglog(W_range, J_values, 'b-', linewidth=2, label='Full solution')
    ax1.loglog(W_range, [flux_DL(W, R) for W in W_range], 'g--', linewidth=2, label='DL: $J^*=1$')
    ax1.loglog(W_range, [flux_SL(W, R) for W in W_range], 'r--', linewidth=2, label=f'SL: $J^*=WR/(1+R)$')

    ax1.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=1, color='gray', linestyle=':', alpha=0.5)

    ax1.set_xlabel('$W$ (-)', fontsize=12)
    ax1.set_ylabel('$J^*$ (-)', fontsize=12)
    ax1.set_title(f'Flux vs Total W (R = {R})\n'
                  f'(Same equation, but W definition differs)', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1e-4, 1e4])
    ax1.set_ylim([1e-5, 2])

    # Panel 2: How W depends on individual contributions
    ax2 = axes[1]

    # For 2-layer: W = W1 + W2 (lines are straight in log-log)
    # For 3-layer: W = (W1+W2)(W2+W3) (more complex)

    W1_range = np.logspace(-2, 2, 50)

    # 2-layer case: W = W1 + W2 (for W2 fixed)
    for W2 in [0.1, 1, 10]:
        W_2layer = W1_range + W2
        ax2.loglog(W1_range, W_2layer, '--', linewidth=2, label=f'2-layer: $W_1$+{W2}')

    # 3-layer case: W = (W1+W2)(W2+W3) (for W2, W3 fixed)
    W3 = 1.0
    for W2 in [0.1, 1, 10]:
        W_3layer = (W1_range + W2) * (W2 + W3)
        ax2.loglog(W1_range, W_3layer, '-', linewidth=2, label=f'3-layer: $W_2$={W2}, $W_3$={W3}')

    ax2.set_xlabel('$W_1$ (-)', fontsize=12)
    ax2.set_ylabel('Total $W$ (-)', fontsize=12)
    ax2.set_title('W Scaling: 2-layer (additive) vs 3-layer (product)', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    # Generate and save basic plots (similar to 2-layer)
    print("Generating regime map (W vs R)...")
    fig1 = plot_regime_map()
    fig1.savefig(os.path.join(FIGS_DIR, 'regime_map.png'), dpi=150, bbox_inches='tight')

    print("Generating flux vs W plot...")
    fig2 = plot_flux_vs_W()
    fig2.savefig(os.path.join(FIGS_DIR, 'flux_vs_W.png'), dpi=150, bbox_inches='tight')

    print("Generating error vs W plot...")
    fig3 = plot_error_1D(R=1.0)
    fig3.savefig(os.path.join(FIGS_DIR, 'error_vs_W.png'), dpi=150, bbox_inches='tight')

    # Generate 3-layer specific plots
    print("Generating W1-W3 regime maps for different W2...")
    fig4 = plot_regime_map_W1W3()
    fig4.savefig(os.path.join(FIGS_DIR, 'regime_map_W1W3.png'), dpi=150, bbox_inches='tight')

    print("Generating middle layer sensitivity analysis...")
    fig5 = plot_middle_layer_sensitivity()
    fig5.savefig(os.path.join(FIGS_DIR, 'middle_layer_sensitivity.png'), dpi=150, bbox_inches='tight')

    print("Generating flux vs individual Wi plots...")
    fig6 = plot_flux_vs_individual_Wi()
    fig6.savefig(os.path.join(FIGS_DIR, 'flux_vs_individual_Wi.png'), dpi=150, bbox_inches='tight')

    print("Generating iso-W contour plots...")
    fig7 = plot_iso_W_contours()
    fig7.savefig(os.path.join(FIGS_DIR, 'iso_W_contours.png'), dpi=150, bbox_inches='tight')

    print("Generating layer dominance analysis...")
    fig8 = plot_layer_dominance_analysis()
    fig8.savefig(os.path.join(FIGS_DIR, 'layer_dominance_analysis.png'), dpi=150, bbox_inches='tight')

    print("Generating 2-layer vs 3-layer comparison...")
    fig9 = plot_comparison_2layer_vs_3layer()
    fig9.savefig(os.path.join(FIGS_DIR, 'comparison_2layer_vs_3layer.png'), dpi=150, bbox_inches='tight')

    print(f"\nAll figures saved to {FIGS_DIR}")
    plt.show()

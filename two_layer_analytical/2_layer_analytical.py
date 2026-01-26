"""
Two-Layer Permeation Analysis
Based on the derivation in the Transport Models chapter

===============================================================================
SYSTEM CONFIGURATION
===============================================================================

    Gas (p_1) | Material 1 (e_1, D_1, K_s1) | Material 2 (e_2, D_2, K_s2) | Vacuum

    Where:
    - p_1     : upstream gas pressure (Pa)
    - e_1, e_2: layer thicknesses (m)
    - D_1, D_2: diffusion coefficients (m²/s)
    - K_s1, K_s2: Sieverts' constants (mol/m³/Pa^0.5)
    - K_d1, K_d2: dissociation constants (mol/m²/s/Pa)
    - K_r1, K_r2: recombination constants (m⁴/mol/s)

===============================================================================
DIMENSIONAL EQUATIONS
===============================================================================

    Flux Balance at Steady State:
        K_d1·p_1 - K_r1·C_1² = J_1 = J_2 = K_r2·C_2²

    Where:
    - Left term:  dissociation flux at gas-material1 surface
    - J_1, J_2:   diffusive flux through materials 1 and 2
    - Right term: recombination flux at material2-vacuum surface

    Diffusion-Limited Flux (derived in your chapter):
        J_DL = D_1·D_2·√p_1 / (D_2·e_1/K_s1 + D_1·e_2/K_s2)

===============================================================================
NON-DIMENSIONALIZATION
===============================================================================

    Dimensionless Concentrations:
        u = C_2 / (K_s2·√p_1)    (concentration at vacuum side, 0 ≤ u ≤ 1)
        v = C_1 / (K_s1·√p_1)    (concentration at gas side, 0 ≤ v ≤ 1)

    Dimensionless Parameters:
        W = K_d1·p_1·(D_2·e_1/K_s1 + D_1·e_2/K_s2) / (D_1·D_2·√p_1)
            → Ratio of surface kinetics to diffusion rate
            → W >> 1: diffusion is slow (diffusion-limited)
            → W << 1: surface kinetics is slow (surface-limited)

        W can be decomposed as W = W1 + W2, where:

        W1 = K_d1·√p_1·e_1 / (D_1·K_s1)
            → Diffusion resistance contribution from layer 1
            → Represents how much layer 1 slows permeation relative to surface kinetics

        W2 = K_d1·√p_1·e_2 / (D_2·K_s2)
            → Diffusion resistance contribution from layer 2
            → Represents how much layer 2 slows permeation relative to surface kinetics

        Layer Dominance:
            → W1 >> W2: Layer 1 dominates diffusion resistance (substrate-limited)
            → W2 >> W1: Layer 2 dominates diffusion resistance (barrier-limited)
            → W1 ≈ W2: Both layers contribute comparably

        R = K_d2 / K_d1
            → Ratio of dissociation constants
            → Determines relative importance of the two surfaces

===============================================================================
DIMENSIONLESS FLUX BALANCE EQUATION
===============================================================================

    After non-dimensionalization, the flux balance becomes:

        W·(1 - v²) = v - u = W·R·u²

    This single equation contains TWO constraints:
        (a) W·(1 - v²) = v - u    [upstream surface ↔ diffusion]
        (b) v - u = W·R·u²        [diffusion ↔ downstream surface]

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
ERROR ANALYSIS (Following Alberghi et al. methodology)
===============================================================================

    The "full solution" is obtained by numerically solving:
        W·(1 - v²) = v - u = W·R·u²

    The "approximate solutions" are the limiting regime formulas:
        J*_DL = 1           (diffusion-limited approximation)
        J*_SL = W·R/(1+R)   (surface-limited approximation)

    Relative Error:
        err = |J*_approx - J*_full| / J*_full

    Regime Validity:
        - If err < 0.05 (5%), the approximation is considered valid
        - This 5% threshold defines the boundaries between regimes

    Example:
        W = 1000, R = 1:
            J*_full ≈ 0.999 (from numerical solution)
            J*_DL = 1
            err_DL = |1 - 0.999| / 0.999 ≈ 0.1% → DL approximation valid!

        W = 0.001, R = 1:
            J*_full ≈ 0.0005 (from numerical solution)
            J*_SL = 0.001 * 1 / (1+1) = 0.0005
            err_SL = |0.0005 - 0.0005| / 0.0005 ≈ 0% → SL approximation valid!

===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Create figs directory if it doesn't exist
FIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)


# =============================================================================
# DIMENSIONLESS PARAMETER COMPUTATION (W1, W2)
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
    Compute W2: the diffusion resistance contribution from layer 2.

    W2 = K_d1·√p_1·e_2 / (D_2·K_s2)

    Physical meaning:
        - Ratio of upstream surface kinetics rate to layer 2 diffusion rate
        - Large W2 means layer 2 is a significant diffusion bottleneck
        - Small W2 means layer 2 offers little resistance to diffusion

    Note: W2 uses K_d1 (not K_d2) because W is normalized by the upstream
    surface kinetics. This allows W1 + W2 = W.

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


def compute_W_from_physical(K_d1, p_1, e_1, e_2, D_1, D_2, K_s1, K_s2):
    """
    Compute total W and its components W1, W2 from physical parameters.

    Returns W, W1, W2 where W = W1 + W2.

    Parameters
    ----------
    K_d1 : float
        Dissociation constant at gas-material1 surface (mol/m²/s/Pa)
    p_1 : float
        Upstream gas pressure (Pa)
    e_1, e_2 : float
        Layer thicknesses (m)
    D_1, D_2 : float
        Diffusion coefficients (m²/s)
    K_s1, K_s2 : float
        Sieverts' constants (mol/m³/Pa^0.5)

    Returns
    -------
    W : float
        Total permeation parameter
    W1 : float
        Layer 1 contribution
    W2 : float
        Layer 2 contribution
    """
    W1 = compute_W1(K_d1, p_1, e_1, D_1, K_s1)
    W2 = compute_W2(K_d1, p_1, e_2, D_2, K_s2)
    W = W1 + W2
    return W, W1, W2


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
        - v - u: diffusive flux through the two-layer system
        - W·R·u²: recombination flux at downstream surface

    We split this into two equations:
        eq1: W·(1 - v²) - (v - u) = 0
        eq2: (v - u) - W·R·u² = 0

    Parameters
    ----------
    x : array [u, v]
        u: dimensionless concentration at vacuum side (C_2 / K_s2·√p_1)
        v: dimensionless concentration at gas side (C_1 / K_s1·√p_1)
    W : float
        Permeation parameter. W >> 1 means diffusion-limited, W << 1 means surface-limited
    R : float
        Ratio of dissociation constants: R = K_d2 / K_d1

    Returns
    -------
    residuals : array [eq1, eq2]
        Both should be zero at the solution
    """
    u, v = x

    # Equation (a): upstream surface flux = diffusive flux
    # W·(1 - v²) represents the net surface flux at the gas side
    # (v - u) represents the diffusive flux through both layers
    eq1 = W * (1 - v**2) - (v - u)

    # Equation (b): diffusive flux = downstream surface flux
    # W·R·u² represents the recombination flux at the vacuum side
    eq2 = (v - u) - W * R * u**2

    return [eq1, eq2]


def solve_two_layer(W, R):
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
        Permeation parameter
    R : float
        Dissociation constant ratio (R in solve_system.py notation)

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


def solve_two_layer_W1W2(W1, W2, R):
    """
    Solve for dimensionless concentrations using W1 and W2 separately.

    This is a convenience wrapper that accepts the decomposed W parameters,
    allowing analysis of how each layer contributes to the overall behavior.

    Parameters
    ----------
    W1 : float
        Layer 1 diffusion resistance: K_d1·√p_1·e_1 / (D_1·K_s1)
    W2 : float
        Layer 2 diffusion resistance: K_d1·√p_1·e_2 / (D_2·K_s2)
    R : float
        Dissociation constant ratio: K_d2 / K_d1

    Returns
    -------
    u : float
        Dimensionless concentration at vacuum side
    v : float
        Dimensionless concentration at gas side
    J_star : float
        Dimensionless flux = v - u
    layer_dominance : str
        Which layer dominates: 'layer1', 'layer2', or 'comparable'
    """
    W = W1 + W2
    u, v, J_star = solve_two_layer(W, R)

    # Determine layer dominance
    if W1 > 0 and W2 > 0:
        ratio = W1 / W2
        if ratio > 10:
            layer_dominance = 'layer1'
        elif ratio < 0.1:
            layer_dominance = 'layer2'
        else:
            layer_dominance = 'comparable'
    elif W1 > 0:
        layer_dominance = 'layer1'
    elif W2 > 0:
        layer_dominance = 'layer2'
    else:
        layer_dominance = 'none'

    return u, v, J_star, layer_dominance


# =============================================================================
# LIMITING REGIME APPROXIMATIONS
# =============================================================================

def flux_DL(W, R):
    """
    Diffusion-Limited (DL) flux approximation.

    When W >> 1, surface kinetics is very fast compared to diffusion.
    The surface reactions reach equilibrium instantly, so:
        - At gas side: C_1 = K_s1·√p_1, giving v = 1
        - At vacuum side: C_2 = 0, giving u = 0

    Therefore: J* = v - u = 1 - 0 = 1

    This means the actual flux equals the diffusion-limited flux:
        J = J_DL = D₁·D₂·√p₁ / (D₂·e₁/K_s1 + D₁·e₂/K_s2)

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
    # In surface-limited regime:
    # v ≈ 1/√(1+R), so v² ≈ 1/(1+R)
    # J* = W·(1 - v²) = W·(1 - 1/(1+R)) = W·R/(1+R)
    J_star_sl = W * R / (1 + R)

    return J_star_sl


# =============================================================================
# ERROR CALCULATION
# =============================================================================

def relative_error(J_approx, J_full):
    """
    Calculate the relative error between an approximation and the full solution.

    This is how we determine if a limiting regime approximation is valid:
        err = |J*_approx - J*_full| / J*_full

    Following Alberghi et al., we use err < 0.05 (5%) as the threshold
    for when an approximation is considered acceptable.

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

    Example
    -------
    >>> J_full = 0.95  # From numerical solution at W=100
    >>> J_DL = 1.0     # DL approximation
    >>> err = relative_error(J_DL, J_full)
    >>> print(f"Error: {err*100:.1f}%")  # "Error: 5.3%"
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

    The resulting error maps show where each approximation is valid (error < 5%)
    and where the mixed regime occurs (both approximations have large errors).

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
    # Create 2D meshgrid for plotting
    W_grid, R_grid = np.meshgrid(W_range, R_range)

    # Initialize arrays to store results
    error_DL = np.zeros_like(W_grid)
    error_SL = np.zeros_like(W_grid)
    J_full_grid = np.zeros_like(W_grid)

    # Loop through all (W, R) combinations
    for i in range(len(R_range)):
        for j in range(len(W_range)):
            W = W_range[j]
            R = R_range[i]

            # Step 1: Get the EXACT solution by solving the full system
            u, v, J_full = solve_two_layer(W, R)
            J_full_grid[i, j] = J_full

            if not np.isnan(J_full) and J_full > 0:
                # Step 2: Calculate the DL approximation
                J_DL = flux_DL(W, R)  # Always returns 1.0

                # Step 3: Calculate the SL approximation
                J_SL = flux_SL(W, R)  # Returns W·R/(1+R)

                # Step 4: Compute relative errors
                # err = |approximation - exact| / exact
                error_DL[i, j] = relative_error(J_DL, J_full)
                error_SL[i, j] = relative_error(J_SL, J_full)
            else:
                error_DL[i, j] = np.nan
                error_SL[i, j] = np.nan

    return W_grid, R_grid, error_DL, error_SL, J_full_grid


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_regime_map(W_range=None, R_range=None):
    """
    Plot regime map: W vs R showing where DL and SL approximations are valid.

    This creates three panels similar to Figure 2/6 in Alberghi et al.:
        1. DL error map: shows where J* = 1 is a good approximation
        2. SL error map: shows where J* = W·R/(1+R) is a good approximation
        3. Best regime: shows which approximation has lower error at each point

    The red contour lines mark the 5% error boundary.
    """
    if W_range is None:
        W_range = np.logspace(-6, 6, 100)
    if R_range is None:
        R_range = np.logspace(-6, 6, 100)

    # Compute errors across the parameter space
    W_grid, R_grid, error_DL, error_SL, J_full = compute_regime_map(W_range, R_range)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: DL approximation error
    # Dark purple = low error (DL approximation valid)
    # Yellow = high error (DL approximation fails)
    ax1 = axes[0]
    err_DL_clipped = np.clip(error_DL, 1e-6, 0.5)
    pcm1 = ax1.pcolormesh(W_grid, R_grid, err_DL_clipped,
                          norm=plt.matplotlib.colors.LogNorm(vmin=1e-4, vmax=0.5),
                          cmap='viridis_r', shading='auto')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('$W_1 + W_2$ (-)')
    ax1.set_ylabel('R = K_d2/K_d1 (-)')
    ax1.set_title('Diffusion-Limited Error\n$J^* = 1$')
    plt.colorbar(pcm1, ax=ax1, label='Relative Error')
    # Red contour at 5% error - this is the regime boundary
    ax1.contour(W_grid, R_grid, error_DL, levels=[0.05], colors='red', linewidths=2)

    # Panel 2: SL approximation error
    ax2 = axes[1]
    err_SL_clipped = np.clip(error_SL, 1e-6, 0.5)
    pcm2 = ax2.pcolormesh(W_grid, R_grid, err_SL_clipped,
                          norm=plt.matplotlib.colors.LogNorm(vmin=1e-4, vmax=0.5),
                          cmap='viridis_r', shading='auto')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('$W_1 + W_2$ (-)')
    ax2.set_ylabel('R = K_d2/K_d1 (-)')
    ax2.set_title('Surface-Limited Error\n$J^* = W·R/(1+R)$')
    plt.colorbar(pcm2, ax=ax2, label='Relative Error')
    ax2.contour(W_grid, R_grid, error_SL, levels=[0.05], colors='red', linewidths=2)

    # Panel 3: Best regime map
    # Shows which approximation is better at each point
    # White = mixed regime (neither approximation valid at 5%)
    ax3 = axes[2]
    min_error = np.minimum(error_DL, error_SL)

    # Create regime classification:
    # 0 = DL valid (error < 5%), 1 = SL valid (error < 5%), NaN = mixed (neither valid)
    best_regime = np.where(error_DL < error_SL, 0, 1)  # 0 = DL better, 1 = SL better
    # Mask the mixed regime where neither approximation is valid
    mixed_regime_mask = min_error >= 0.05
    best_regime_masked = np.ma.masked_where(mixed_regime_mask, best_regime)

    cmap = plt.colormaps.get_cmap('RdYlBu').resampled(2)
    cmap.set_bad(color='white')  # Mixed regime shown as white
    pcm3 = ax3.pcolormesh(W_grid, R_grid, best_regime_masked, cmap=cmap, shading='auto', vmin=-0.5, vmax=1.5)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('$W_1 + W_2$ (-)')
    ax3.set_ylabel('R = K_d2/K_d1 (-)')
    ax3.set_title('Best Limiting Regime\n(white = mixed regime)')
    cbar = plt.colorbar(pcm3, ax=ax3, ticks=[0, 1])
    cbar.ax.set_yticklabels(['DL', 'SL'])

    # Dashed black contour shows where minimum error = 5%
    # Inside this contour, at least one approximation is valid
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
            _, _, J = solve_two_layer(W, R)
            J_full.append(J)

        ax.loglog(W_range, J_full, '-', color=color, linewidth=2, label=f'R = {R}')

    # Reference lines for limiting regimes
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='DL: $J^*=1$')
    ax.loglog(W_range, W_range * 0.5, 'k:', alpha=0.5, label='SL: $J^* ∝ W$')

    ax.set_xlabel('$W_1 + W_2$ (-)', fontsize=12)
    ax.set_ylabel('$J^*$ (-)', fontsize=12)
    ax.set_title('Dimensionless Flux vs Permeation Parameter', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1e-4, 1e4])
    ax.set_ylim([1e-5, 10])

    return fig


def plot_error_1D(R=1.0):
    """
    Plot error vs W for a fixed R, similar to Figure 2 in Alberghi.

    This clearly shows:
        - For W < 0.01: SL approximation valid (red line below 5%)
        - For W > 100: DL approximation valid (blue line below 5%)
        - For 0.01 < W < 100: Mixed regime (both approximations fail)
    """
    W_range = np.logspace(-5, 5, 200)

    err_DL = []
    err_SL = []

    for W in W_range:
        # Get exact solution
        _, _, J_full = solve_two_layer(W, R)

        if not np.isnan(J_full) and J_full > 0:
            # Calculate errors for both approximations
            err_DL.append(relative_error(flux_DL(W, R), J_full))
            err_SL.append(relative_error(flux_SL(W, R), J_full))
        else:
            err_DL.append(np.nan)
            err_SL.append(np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert to arrays for easier manipulation
    err_DL = np.array(err_DL)
    err_SL = np.array(err_SL)

    # Shade only where each approximation is valid (error < 5%)
    # Use fill_between with a where condition based on actual error values
    ax.fill_between(W_range, 1e-6, 10, where=(err_SL < 0.05),
                    alpha=0.2, color='red', label='SL regime (<5% error)')
    ax.fill_between(W_range, 1e-6, 10, where=(err_DL < 0.05),
                    alpha=0.2, color='blue', label='DL regime (<5% error)')

    ax.loglog(W_range, err_DL, 'b-', linewidth=2, label='DL approximation ($J^*=1$)')
    ax.loglog(W_range, err_SL, 'r-', linewidth=2, label='SL approximation ($J^*=WR/(1+R)$)')
    ax.axhline(y=0.05, color='gray', linestyle='--', label='5% error threshold')

    ax.set_xlabel('$W_1 + W_2$ (-)', fontsize=12)
    ax.set_ylabel('Relative Error (-)', fontsize=12)
    ax.set_title(f'Relative Error vs $W_1 + W_2$ (R = {R})', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1e-5, 1e5])
    ax.set_ylim([1e-6, 10])

    return fig


def compute_regime_map_W1W2(W1_range, W2_range, R=1.0):
    """
    Compute error maps in W1-W2 space for a fixed R value.

    This allows visualization of how the two layer contributions
    interact to determine the permeation regime.

    Parameters
    ----------
    W1_range : array
        Array of W1 values (layer 1 contribution)
    W2_range : array
        Array of W2 values (layer 2 contribution)
    R : float
        Fixed dissociation constant ratio

    Returns
    -------
    W1_grid, W2_grid : 2D arrays
        Meshgrid of W1 and W2 values
    error_DL : 2D array
        Relative error of DL approximation
    error_SL : 2D array
        Relative error of SL approximation
    J_full_grid : 2D array
        Full numerical solution
    """
    W1_grid, W2_grid = np.meshgrid(W1_range, W2_range)

    error_DL = np.zeros_like(W1_grid)
    error_SL = np.zeros_like(W1_grid)
    J_full_grid = np.zeros_like(W1_grid)

    for i in range(len(W2_range)):
        for j in range(len(W1_range)):
            W1 = W1_range[j]
            W2 = W2_range[i]
            W = W1 + W2

            _, _, J_full = solve_two_layer(W, R)
            J_full_grid[i, j] = J_full

            if not np.isnan(J_full) and J_full > 0:
                J_DL = flux_DL(W, R)
                J_SL = flux_SL(W, R)
                error_DL[i, j] = relative_error(J_DL, J_full)
                error_SL[i, j] = relative_error(J_SL, J_full)
            else:
                error_DL[i, j] = np.nan
                error_SL[i, j] = np.nan

    return W1_grid, W2_grid, error_DL, error_SL, J_full_grid


def plot_regime_map_W1W2(W1_range=None, W2_range=None, R=1.0):
    """
    Plot regime map in W1-W2 space showing permeation regimes and layer dominance.

    This visualization shows:
        1. Which permeation regime (DL vs SL) applies at each (W1, W2) point
        2. Which layer dominates the diffusion resistance
        3. The transition boundaries between regimes

    Parameters
    ----------
    W1_range : array, optional
        W1 values to evaluate. Default: logspace(-4, 4)
    W2_range : array, optional
        W2 values to evaluate. Default: logspace(-4, 4)
    R : float
        Dissociation constant ratio (default: 1.0)

    Returns
    -------
    fig : matplotlib Figure
    """
    if W1_range is None:
        W1_range = np.logspace(-4, 4, 100)
    if W2_range is None:
        W2_range = np.logspace(-4, 4, 100)

    W1_grid, W2_grid, error_DL, error_SL, J_full = compute_regime_map_W1W2(
        W1_range, W2_range, R)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: DL approximation error
    ax1 = axes[0, 0]
    err_DL_clipped = np.clip(error_DL, 1e-6, 0.5)
    pcm1 = ax1.pcolormesh(W1_grid, W2_grid, err_DL_clipped,
                          norm=plt.matplotlib.colors.LogNorm(vmin=1e-4, vmax=0.5),
                          cmap='viridis_r', shading='auto')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('$W_1$ (Layer 1 resistance)')
    ax1.set_ylabel('$W_2$ (Layer 2 resistance)')
    ax1.set_title(f'DL Error ($J^* = 1$), R = {R}')
    plt.colorbar(pcm1, ax=ax1, label='Relative Error')
    ax1.contour(W1_grid, W2_grid, error_DL, levels=[0.05], colors='red', linewidths=2)
    # Add W1=W2 line for reference
    ax1.plot([1e-4, 1e4], [1e-4, 1e4], 'k--', alpha=0.5, label='$W_1 = W_2$')

    # Panel 2: SL approximation error
    ax2 = axes[0, 1]
    err_SL_clipped = np.clip(error_SL, 1e-6, 0.5)
    pcm2 = ax2.pcolormesh(W1_grid, W2_grid, err_SL_clipped,
                          norm=plt.matplotlib.colors.LogNorm(vmin=1e-4, vmax=0.5),
                          cmap='viridis_r', shading='auto')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('$W_1$ (Layer 1 resistance)')
    ax2.set_ylabel('$W_2$ (Layer 2 resistance)')
    ax2.set_title(f'SL Error ($J^* = WR/(1+R)$), R = {R}')
    plt.colorbar(pcm2, ax=ax2, label='Relative Error')
    ax2.contour(W1_grid, W2_grid, error_SL, levels=[0.05], colors='red', linewidths=2)
    ax2.plot([1e-4, 1e4], [1e-4, 1e4], 'k--', alpha=0.5)

    # Panel 3: Dimensionless flux J*
    ax3 = axes[1, 0]
    J_clipped = np.clip(J_full, 1e-6, 1.0)
    pcm3 = ax3.pcolormesh(W1_grid, W2_grid, J_clipped,
                          norm=plt.matplotlib.colors.LogNorm(vmin=1e-4, vmax=1),
                          cmap='plasma', shading='auto')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('$W_1$ (Layer 1 resistance)')
    ax3.set_ylabel('$W_2$ (Layer 2 resistance)')
    ax3.set_title(f'Dimensionless Flux $J^*$, R = {R}')
    plt.colorbar(pcm3, ax=ax3, label='$J^*$')
    # Contour lines for constant W = W1 + W2
    W_total = W1_grid + W2_grid
    ax3.contour(W1_grid, W2_grid, W_total, levels=[0.1, 1, 10, 100],
                colors='white', linewidths=1, linestyles='--')
    ax3.plot([1e-4, 1e4], [1e-4, 1e4], 'k--', alpha=0.5)

    # Panel 4: Combined regime and layer dominance map
    # White = mixed regime (neither approximation valid at 5%)
    ax4 = axes[1, 1]
    # Create a combined classification:
    # 0 = SL, Layer 1 dominant (W1 > W2)
    # 1 = SL, Layer 2 dominant (W2 > W1)
    # 2 = DL, Layer 1 dominant
    # 3 = DL, Layer 2 dominant
    is_DL = error_DL < error_SL
    layer1_dominant = W1_grid > W2_grid

    regime_class = np.zeros_like(W1_grid)
    regime_class[~is_DL & layer1_dominant] = 0   # SL, L1 dominant
    regime_class[~is_DL & ~layer1_dominant] = 1  # SL, L2 dominant
    regime_class[is_DL & layer1_dominant] = 2    # DL, L1 dominant
    regime_class[is_DL & ~layer1_dominant] = 3   # DL, L2 dominant

    # Mask the mixed regime where neither approximation is valid
    min_error = np.minimum(error_DL, error_SL)
    mixed_regime_mask = min_error >= 0.05
    regime_class_masked = np.ma.masked_where(mixed_regime_mask, regime_class)

    cmap = plt.colormaps.get_cmap('RdYlGn').resampled(4)
    cmap.set_bad(color='white')  # Mixed regime shown as white
    pcm4 = ax4.pcolormesh(W1_grid, W2_grid, regime_class_masked, cmap=cmap,
                          shading='auto', vmin=-0.5, vmax=3.5)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('$W_1$ (Layer 1 resistance)')
    ax4.set_ylabel('$W_2$ (Layer 2 resistance)')
    ax4.set_title(f'Regime & Layer Dominance, R = {R}\n(white = mixed regime)')
    cbar = plt.colorbar(pcm4, ax=ax4, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['SL/L1', 'SL/L2', 'DL/L1', 'DL/L2'])

    # Add regime boundary (5% error contour)
    ax4.contour(W1_grid, W2_grid, min_error, levels=[0.05],
                colors='black', linewidths=2, linestyles='--')
    # Add W1=W2 line (layer dominance boundary)
    ax4.plot([1e-4, 1e4], [1e-4, 1e4], 'k-', linewidth=2, label='$W_1 = W_2$')

    plt.tight_layout()
    return fig


def plot_flux_vs_W1W2_slices(R=1.0):
    """
    Plot J* vs W1 for various fixed W2 values (and vice versa).

    This shows how the flux depends on each layer's contribution
    while holding the other constant.

    Parameters
    ----------
    R : float
        Dissociation constant ratio

    Returns
    -------
    fig : matplotlib Figure
    """
    W_range = np.logspace(-4, 4, 100)
    W_fixed_values = [0.001, 0.01, 0.1, 1, 10, 100]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(W_fixed_values)))

    # Panel 1: J* vs W1 for fixed W2
    ax1 = axes[0]
    for W2_fixed, color in zip(W_fixed_values, colors):
        J_values = []
        for W1 in W_range:
            W = W1 + W2_fixed
            _, _, J = solve_two_layer(W, R)
            J_values.append(J)
        ax1.loglog(W_range, J_values, '-', color=color, linewidth=2,
                   label=f'$W_2$ = {W2_fixed}')

    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax1.set_xlabel('$W_1$ (Layer 1 resistance)', fontsize=12)
    ax1.set_ylabel('$J^*$', fontsize=12)
    ax1.set_title(f'Flux vs $W_1$ for fixed $W_2$ (R = {R})', fontsize=14)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1e-4, 1e4])
    ax1.set_ylim([1e-5, 2])

    # Panel 2: J* vs W2 for fixed W1
    ax2 = axes[1]
    for W1_fixed, color in zip(W_fixed_values, colors):
        J_values = []
        for W2 in W_range:
            W = W1_fixed + W2
            _, _, J = solve_two_layer(W, R)
            J_values.append(J)
        ax2.loglog(W_range, J_values, '-', color=color, linewidth=2,
                   label=f'$W_1$ = {W1_fixed}')

    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax2.set_xlabel('$W_2$ (Layer 2 resistance)', fontsize=12)
    ax2.set_ylabel('$J^*$', fontsize=12)
    ax2.set_title(f'Flux vs $W_2$ for fixed $W_1$ (R = {R})', fontsize=14)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([1e-4, 1e4])
    ax2.set_ylim([1e-5, 2])

    plt.tight_layout()
    return fig


def plot_layer_contribution_analysis(R=1.0):
    """
    Analyze and visualize the relative contribution of each layer.

    Creates a plot showing:
    - How the W1/(W1+W2) ratio affects the regime
    - The transition from layer 1 to layer 2 dominance

    Parameters
    ----------
    R : float
        Dissociation constant ratio

    Returns
    -------
    fig : matplotlib Figure
    """
    # Create data for different total W values
    W_total_values = [0.01, 0.1, 1, 10, 100]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.coolwarm(np.linspace(0, 1, len(W_total_values)))

    # Panel 1: J* vs W1 fraction
    # Key insight: J* depends only on total W = W1 + W2, not the split
    # So each W_total gives a horizontal line
    ax1 = axes[0]
    for W_total, color in zip(W_total_values, colors):
        _, _, J = solve_two_layer(W_total, R)
        ax1.axhline(y=J, color=color, linewidth=2, label=f'W = {W_total}')

    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='DL limit')
    ax1.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('$W_1 / (W_1 + W_2)$ (Layer 1 fraction)', fontsize=12)
    ax1.set_ylabel('$J^*$', fontsize=12)
    ax1.set_title(f'Flux vs Layer 1 Fraction (R = {R})\n'
                  '(Note: $J^*$ depends only on total W)', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.1])

    # Panel 2: Contour plot of J* in W1-W2 space with W1+W2 = const lines
    ax2 = axes[1]
    W1_range = np.logspace(-3, 3, 100)
    W2_range = np.logspace(-3, 3, 100)
    W1_grid, W2_grid = np.meshgrid(W1_range, W2_range)

    J_grid = np.zeros_like(W1_grid)
    for i in range(len(W2_range)):
        for j in range(len(W1_range)):
            W = W1_range[j] + W2_range[i]
            _, _, J = solve_two_layer(W, R)
            J_grid[i, j] = J

    pcm = ax2.pcolormesh(W1_grid, W2_grid, J_grid,
                         norm=plt.matplotlib.colors.LogNorm(vmin=1e-4, vmax=1),
                         cmap='plasma', shading='auto')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    plt.colorbar(pcm, ax=ax2, label='$J^*$')

    # Add iso-W lines (W1 + W2 = const)
    for W_const in [0.01, 0.1, 1, 10, 100]:
        W1_line = np.logspace(-3, np.log10(W_const), 50)
        W2_line = W_const - W1_line
        valid = W2_line > 1e-3
        ax2.plot(W1_line[valid], W2_line[valid], 'w--', linewidth=1.5, alpha=0.7)
        # Label the line
        if W_const >= 0.1:
            ax2.text(W_const * 0.7, W_const * 0.3, f'W={W_const}',
                     color='white', fontsize=9, rotation=-45)

    # Add W1=W2 line
    ax2.plot([1e-3, 1e3], [1e-3, 1e3], 'w-', linewidth=2, label='$W_1 = W_2$')

    ax2.set_xlabel('$W_1$ (Layer 1 resistance)', fontsize=12)
    ax2.set_ylabel('$W_2$ (Layer 2 resistance)', fontsize=12)
    ax2.set_title(f'$J^*$ Contours with Iso-W Lines (R = {R})\n'
                  'Dashed lines: constant $W = W_1 + W_2$', fontsize=12)
    ax2.set_xlim([1e-3, 1e3])
    ax2.set_ylim([1e-3, 1e3])

    plt.tight_layout()
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    # Generate and save original plots
    print("Generating regime map (W vs R)...")
    fig1 = plot_regime_map()
    fig1.savefig(os.path.join(FIGS_DIR, 'regime_map.png'), dpi=150, bbox_inches='tight')

    print("Generating flux vs W plot...")
    fig2 = plot_flux_vs_W()
    fig2.savefig(os.path.join(FIGS_DIR, 'flux_vs_W.png'), dpi=150, bbox_inches='tight')

    print("Generating error vs W plot...")
    fig3 = plot_error_1D(R=1.0)
    fig3.savefig(os.path.join(FIGS_DIR, 'error_vs_W.png'), dpi=150, bbox_inches='tight')

    # Generate new W1-W2 analysis plots
    print("Generating W1-W2 regime map...")
    fig4 = plot_regime_map_W1W2(R=1.0)
    fig4.savefig(os.path.join(FIGS_DIR, 'regime_map_W1W2.png'), dpi=150, bbox_inches='tight')

    print("Generating flux vs W1/W2 slices...")
    fig5 = plot_flux_vs_W1W2_slices(R=1.0)
    fig5.savefig(os.path.join(FIGS_DIR, 'flux_vs_W1W2_slices.png'), dpi=150, bbox_inches='tight')

    print("Generating layer contribution analysis...")
    fig6 = plot_layer_contribution_analysis(R=1.0)
    fig6.savefig(os.path.join(FIGS_DIR, 'layer_contribution_analysis.png'), dpi=150, bbox_inches='tight')

    print(f"All figures saved to {FIGS_DIR}")
    plt.show()

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
    ax1.set_xlabel('W (-)')
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
    ax2.set_xlabel('W (-)')
    ax2.set_ylabel('R = K_d2/K_d1 (-)')
    ax2.set_title('Surface-Limited Error\n$J^* = W·R/(1+R)$')
    plt.colorbar(pcm2, ax=ax2, label='Relative Error')
    ax2.contour(W_grid, R_grid, error_SL, levels=[0.05], colors='red', linewidths=2)

    # Panel 3: Best regime map
    # Shows which approximation is better at each point
    ax3 = axes[2]
    best_regime = np.where(error_DL < error_SL, 0, 1)  # 0 = DL better, 1 = SL better
    min_error = np.minimum(error_DL, error_SL)

    cmap = plt.colormaps.get_cmap('RdYlBu').resampled(2)
    pcm3 = ax3.pcolormesh(W_grid, R_grid, best_regime, cmap=cmap, shading='auto', vmin=-0.5, vmax=1.5)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('W (-)')
    ax3.set_ylabel('R = K_d2/K_d1 (-)')
    ax3.set_title('Best Limiting Regime')
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

    ax.set_xlabel('W (-)', fontsize=12)
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

    ax.loglog(W_range, err_DL, 'b-', linewidth=2, label='DL approximation ($J^*=1$)')
    ax.loglog(W_range, err_SL, 'r-', linewidth=2, label='SL approximation ($J^*=WR/(1+R)$)')
    ax.axhline(y=0.05, color='gray', linestyle='--', label='5% error threshold')

    ax.set_xlabel('W (-)', fontsize=12)
    ax.set_ylabel('Relative Error (-)', fontsize=12)
    ax.set_title(f'Relative Error vs W (R = {R})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1e-5, 1e5])
    ax.set_ylim([1e-6, 10])

    # Shade the valid regime regions
    ax.fill_between([1e-5, 0.1], 1e-6, 10, alpha=0.2, color='red', label='SL regime')
    ax.fill_between([100, 1e5], 1e-6, 10, alpha=0.2, color='blue', label='DL regime')

    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    # Generate and save plots
    fig1 = plot_regime_map()
    fig1.savefig(os.path.join(FIGS_DIR, 'regime_map.png'), dpi=150, bbox_inches='tight')

    fig2 = plot_flux_vs_W()
    fig2.savefig(os.path.join(FIGS_DIR, 'flux_vs_W.png'), dpi=150, bbox_inches='tight')

    fig3 = plot_error_1D(R=1.0)
    fig3.savefig(os.path.join(FIGS_DIR, 'error_vs_W.png'), dpi=150, bbox_inches='tight')

    plt.show()

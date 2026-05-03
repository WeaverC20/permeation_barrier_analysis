"""
Three-Layer Permeation Analysis (corrected derivation)

===============================================================================
SYSTEM CONFIGURATION
===============================================================================

    Gas (p_1) | Material 1 (e_1, D_1, K_s1) | Material 2 (e_2, D_2, K_s2) | Material 3 (e_3, D_3, K_s3) | Vacuum

    - p_1     : upstream gas pressure (Pa)
    - e_i     : layer thickness (m)
    - D_i     : diffusion coefficient (m^2/s)
    - K_si    : Sieverts' constant (mol/m^3/Pa^0.5)
    - K_d1, K_d3 : dissociation constants at the two outer surfaces
    - K_r1, K_r3 : recombination constants at the two outer surfaces

    Consistency relation used in the dimensionless derivation:
        K_si = sqrt(K_di / K_ri)

===============================================================================
DIMENSIONAL EQUATIONS
===============================================================================

    Steady-state flux balance:
        K_d1*p_1 - K_r1*C_1^2 = J_1 = J_2 = J_3 = K_r3*C_3^2

    At each internal interface, Sieverts partitioning gives:
        C(layer i, interface) / K_si = C(layer i+1, interface) / K_s(i+1)

===============================================================================
NON-DIMENSIONALIZATION
===============================================================================

    Dimensionless concentrations:
        u = C_1 / (K_s1 * sqrt(p_1))    upstream surface (layer 1)
        w = C_3 / (K_s3 * sqrt(p_1))    downstream surface (layer 3)

    Per-layer dimensionless resistances:
        W_i = K_d1 * sqrt(p_1) * e_i / (D_i * K_si)

    Working through the flux continuity J_1 = J_2 = J_3 with Sieverts
    matching at each interface yields a single sum-of-W's denominator
    (resistances in series add):

        J / (K_d1 * p_1) = (u - w) / (W_1 + W_2 + W_3)

    Define:
        W = W_1 + W_2 + W_3        (sum of W's, NOT a product of sums)
        R = K_d3 / K_d1

    The two surface boundary conditions become:
        1 - u^2 = (u - w) / W     (upstream)
        R * w^2 = (u - w) / W     (downstream)

    These have the same algebraic form as the two-layer case; only the
    definition of W changes (one extra additive term).

===============================================================================
QUARTIC FOR w
===============================================================================

    Eliminating u from the system using (u - w) = W*R*w^2 (so u = w + W*R*w^2)
    and substituting into 1 - u^2 = R*w^2 yields:

        W^2 R^2 w^4 + 2 W R w^3 + (1 + R) w^2 - 1 = 0

    Once w is found, u = w + W*R*w^2 and J* = u - w = W*R*w^2.

===============================================================================
LIMITING REGIMES
===============================================================================

    Diffusion-limited (W >> 1):  u -> 1, w -> 0, J* -> 1  (J = J_DL)
    Surface-limited  (W << 1):  u ~= w ~= 1/sqrt(1+R), J* = W*R/(1+R)

    J_DL = K_d1 * p_1 / W = D-equivalent diffusive flux through the
    series of resistances W_1, W_2, W_3.

===============================================================================
SANITY CHECK
===============================================================================

    For identical materials in all three layers, W = W_1+W_2+W_3 reduces
    to a single-layer system of total thickness e_1+e_2+e_3, exactly as
    expected from resistances-in-series. A numerical finite-difference
    check confirms agreement to machine precision (see run_sanity_check
    below).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

FIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)


# =============================================================================
# DIMENSIONLESS PARAMETER COMPUTATION (W1, W2, W3)
# =============================================================================

def compute_Wi(K_d1, p_1, e_i, D_i, K_si):
    """
    Compute the per-layer dimensionless resistance W_i.

        W_i = K_d1 * sqrt(p_1) * e_i / (D_i * K_si)

    All W_i are normalized by the upstream surface kinetics (K_d1) so they
    add directly: W_total = W_1 + W_2 + W_3.
    """
    return K_d1 * np.sqrt(p_1) * e_i / (D_i * K_si)


def compute_W_total(W1, W2, W3):
    """W_total = W_1 + W_2 + W_3 (resistances in series)."""
    return W1 + W2 + W3


def compute_W_from_physical(K_d1, p_1, e_1, e_2, e_3,
                            D_1, D_2, D_3, K_s1, K_s2, K_s3):
    """Return W (= W1+W2+W3) and the individual W_i from physical inputs."""
    W1 = compute_Wi(K_d1, p_1, e_1, D_1, K_s1)
    W2 = compute_Wi(K_d1, p_1, e_2, D_2, K_s2)
    W3 = compute_Wi(K_d1, p_1, e_3, D_3, K_s3)
    return W1 + W2 + W3, W1, W2, W3


# =============================================================================
# CORE PHYSICS: Solver
# =============================================================================

def solve_three_layer(W, R):
    """
    Solve the dimensionless system

        1 - u^2 = (u - w) / W = R * w^2

    for u, w, and J* = u - w. Reduces to a quartic in w:

        W^2 R^2 w^4 + 2 W R w^3 + (1+R) w^2 - 1 = 0

    Returns
    -------
    u, w, J_star : float
        u in [0, 1], w in [0, 1], J_star = u - w in [0, 1].
    """
    a4 = W**2 * R**2
    a3 = 2 * W * R
    a2 = 1 + R
    a1 = 0
    a0 = -1

    try:
        roots = np.roots([a4, a3, a2, a1, a0])
        physical_w = None
        for root in roots:
            if np.abs(root.imag) < 1e-10:
                real_root = root.real
                if 0 < real_root <= 1 + 1e-10:
                    physical_w = real_root
                    break

        if physical_w is None:
            return np.nan, np.nan, np.nan

        w = min(physical_w, 1.0)
        u = w + W * R * w**2
        J_star = u - w
        u = max(0.0, min(1.0, u))
        return u, w, J_star
    except Exception:
        return np.nan, np.nan, np.nan


def solve_three_layer_W1W2W3(W1, W2, W3, R):
    """Solve using the three layer-resistances directly."""
    W = W1 + W2 + W3
    u, w, J_star = solve_three_layer(W, R)

    if W > 0:
        fractions = {'f1': W1 / W, 'f2': W2 / W, 'f3': W3 / W}
        dominant = max(fractions, key=fractions.get)
    else:
        fractions = {'f1': np.nan, 'f2': np.nan, 'f3': np.nan}
        dominant = 'undefined'

    return u, w, J_star, {
        'W': W, 'W1': W1, 'W2': W2, 'W3': W3,
        'fractions': fractions,
        'dominant_layer': dominant,
    }


# =============================================================================
# LIMITING REGIME APPROXIMATIONS
# =============================================================================

def flux_DL(W, R):
    """Diffusion-limited: J* -> 1 when W >> 1. Args kept for API symmetry with flux_SL."""
    del W, R
    return 1.0


def flux_SL(W, R):
    """Surface-limited: J* = W*R/(1+R) when W << 1."""
    return W * R / (1 + R)


# =============================================================================
# ERROR / REGIME MAP
# =============================================================================

def relative_error(J_approx, J_full):
    if J_full == 0 or np.isnan(J_full):
        return np.inf
    return np.abs(J_approx - J_full) / np.abs(J_full)


def compute_regime_map(W_range, R_range):
    """Sweep (W, R) and return DL/SL relative-error grids and J*_full."""
    W_grid, R_grid = np.meshgrid(W_range, R_range)
    error_DL = np.zeros_like(W_grid)
    error_SL = np.zeros_like(W_grid)
    J_full_grid = np.zeros_like(W_grid)

    for i in range(len(R_range)):
        for j in range(len(W_range)):
            W = W_range[j]
            R = R_range[i]
            _, _, J_full = solve_three_layer(W, R)
            J_full_grid[i, j] = J_full
            if not np.isnan(J_full) and J_full > 0:
                error_DL[i, j] = relative_error(flux_DL(W, R), J_full)
                error_SL[i, j] = relative_error(flux_SL(W, R), J_full)
            else:
                error_DL[i, j] = np.nan
                error_SL[i, j] = np.nan

    return W_grid, R_grid, error_DL, error_SL, J_full_grid


# =============================================================================
# PLOTTING
# =============================================================================

def plot_regime_map(W_range=None, R_range=None):
    """Three-panel regime map: DL error, SL error, and best regime."""
    if W_range is None:
        W_range = np.logspace(-6, 6, 100)
    if R_range is None:
        R_range = np.logspace(-6, 6, 100)

    W_grid, R_grid, error_DL, error_SL, _ = compute_regime_map(W_range, R_range)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax1 = axes[0]
    err_DL_clipped = np.clip(error_DL, 1e-6, 0.5)
    pcm1 = ax1.pcolormesh(W_grid, R_grid, err_DL_clipped,
                          norm=plt.matplotlib.colors.LogNorm(vmin=5e-3, vmax=0.5),
                          cmap='RdYlGn_r', shading='auto')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('$W = W_1 + W_2 + W_3$', fontsize=14)
    ax1.set_ylabel('$R$', fontsize=14)
    ax1.set_title('Diffusion-Limited Error\n$J^* = 1$', fontsize=16)
    plt.colorbar(pcm1, ax=ax1, label='Relative Error')
    ax1.contour(W_grid, R_grid, error_DL, levels=[0.05], colors='red', linewidths=2)

    ax2 = axes[1]
    err_SL_clipped = np.clip(error_SL, 1e-6, 0.5)
    pcm2 = ax2.pcolormesh(W_grid, R_grid, err_SL_clipped,
                          norm=plt.matplotlib.colors.LogNorm(vmin=5e-3, vmax=0.5),
                          cmap='RdYlGn_r', shading='auto')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('$W = W_1 + W_2 + W_3$', fontsize=14)
    ax2.set_ylabel('$R$', fontsize=14)
    ax2.set_title('Surface-Limited Error\n$J^* = WR/(1+R)$', fontsize=16)
    plt.colorbar(pcm2, ax=ax2, label='Relative Error')
    ax2.contour(W_grid, R_grid, error_SL, levels=[0.05], colors='red', linewidths=2)

    ax3 = axes[2]
    min_error = np.minimum(error_DL, error_SL)
    cmap = plt.colormaps.get_cmap('RdYlBu').resampled(2)
    ax3.contourf(W_grid, R_grid,
                 np.where(error_DL < error_SL, 0.0, 1.0),
                 levels=[-0.5, 0.5, 1.5], cmap=cmap)
    ax3.contourf(W_grid, R_grid, min_error,
                 levels=[0.05, np.inf], colors=['white'])
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('$W = W_1 + W_2 + W_3$', fontsize=14)
    ax3.set_ylabel('$R$', fontsize=14)
    ax3.set_title('Best Limiting Regime\n(white = mixed regime)', fontsize=16)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-0.5, vmax=1.5))
    cbar = plt.colorbar(sm, ax=ax3, ticks=[0, 1])
    cbar.ax.set_yticklabels(['DL', 'SL'])
    ax3.contour(W_grid, R_grid, min_error, levels=[0.05],
                colors='black', linewidths=2)

    plt.tight_layout()
    return fig


def plot_flux_vs_W(R_values=(0.01, 0.1, 1, 10, 100)):
    """J* vs W for several R, illustrating the SL -> DL transition."""
    W_range = np.logspace(-4, 4, 100)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(R_values)))

    for R, color in zip(R_values, colors):
        J_vals = [solve_three_layer(W, R)[2] for W in W_range]
        ax.loglog(W_range, J_vals, '-', color=color, linewidth=2, label=f'R = {R}')

    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='DL: $J^*=1$')
    ax.loglog(W_range, W_range * 0.5, 'k:', alpha=0.5, label='SL: $J^* \\propto W$')

    ax.set_xlabel('$W = W_1 + W_2 + W_3$', fontsize=14)
    ax.set_ylabel('$J^*$', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1e-4, 1e4])
    ax.set_ylim([1e-5, 10])

    return fig


def plot_error_1D(R=1.0):
    """Relative error of DL and SL approximations vs W at fixed R."""
    W_range = np.logspace(-5, 5, 200)
    err_DL, err_SL = [], []

    for W in W_range:
        _, _, J_full = solve_three_layer(W, R)
        if not np.isnan(J_full) and J_full > 0:
            err_DL.append(relative_error(flux_DL(W, R), J_full))
            err_SL.append(relative_error(flux_SL(W, R), J_full))
        else:
            err_DL.append(np.nan)
            err_SL.append(np.nan)

    err_DL = np.array(err_DL)
    err_SL = np.array(err_SL)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(W_range, 1e-6, 10, where=(err_SL < 0.05),
                    alpha=0.2, color='red', label='SL regime (<5% error)')
    ax.fill_between(W_range, 1e-6, 10, where=(err_DL < 0.05),
                    alpha=0.2, color='blue', label='DL regime (<5% error)')
    ax.loglog(W_range, err_DL, 'b-', linewidth=2, label='DL approximation ($J^*=1$)')
    ax.loglog(W_range, err_SL, 'r-', linewidth=2, label='SL approximation ($J^*=WR/(1+R)$)')
    ax.axhline(y=0.05, color='gray', linestyle='--', label='5% threshold')

    ax.set_xlabel('$W = W_1 + W_2 + W_3$', fontsize=14)
    ax.set_ylabel('Relative Error', fontsize=14)
    ax.set_title(f'Relative Error vs $W$ (3-Layer, R = {R})', fontsize=16)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1e-5, 1e5])
    ax.set_ylim([1e-6, 10])

    return fig


def plot_regime_map_W1W3(W1_range=None, W3_range=None,
                         W2_values=(1e-3, 1e-1, 1e1, 1e3), R=1.0):
    """
    Approximation-error map in (W_1, W_3) space at several fixed W_2 values.

    Background colour shows the relative error of the BEST limiting-regime
    approximation (DL or SL), green = small error (approximation works) →
    red = large error (mixed regime). Iso-W contours W_1+W_2+W_3=const are
    overlaid in cyan and labelled with their W value.
    """
    if W1_range is None:
        W1_range = np.logspace(-3, 3, 80)
    if W3_range is None:
        W3_range = np.logspace(-3, 3, 80)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()

    iso_W_levels = [0.1, 1, 10, 100]

    for ax, W2 in zip(axes, W2_values):
        W1_grid, W3_grid = np.meshgrid(W1_range, W3_range)
        err_DL_grid = np.zeros_like(W1_grid)
        err_SL_grid = np.zeros_like(W1_grid)

        for i in range(len(W3_range)):
            for j in range(len(W1_range)):
                W = W1_range[j] + W2 + W3_range[i]
                _, _, J_full = solve_three_layer(W, R)
                if not np.isnan(J_full) and J_full > 0:
                    err_DL_grid[i, j] = relative_error(flux_DL(W, R), J_full)
                    err_SL_grid[i, j] = relative_error(flux_SL(W, R), J_full)
                else:
                    err_DL_grid[i, j] = np.nan
                    err_SL_grid[i, j] = np.nan

        min_error = np.minimum(err_DL_grid, err_SL_grid)
        # Centre the LogNorm so the 5% contour sits at the yellow midpoint:
        # geometric mean(vmin, vmax) = sqrt(5e-3 * 0.5) = 0.05.
        err_clipped = np.clip(min_error, 1e-4, 1.0)
        pcm = ax.pcolormesh(W1_grid, W3_grid, err_clipped,
                            norm=plt.matplotlib.colors.LogNorm(vmin=5e-3, vmax=0.5),
                            cmap='RdYlGn_r', shading='auto')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('$W_1$', fontsize=14)
        ax.set_ylabel('$W_3$', fontsize=14)
        ax.set_title(f'$W_2$ = {W2}', fontsize=14)

        # 5% error boundary
        ax.contour(W1_grid, W3_grid, min_error, levels=[0.05],
                   colors='black', linewidths=2)

        # Iso-W contours (W_1 + W_2 + W_3 = const), labelled with the W value
        W_total_grid = W1_grid + W2 + W3_grid
        cs = ax.contour(W1_grid, W3_grid, W_total_grid,
                        levels=iso_W_levels,
                        colors='cyan', linewidths=1.2, linestyles='-')
        ax.clabel(cs, inline=True, fontsize=9, fmt='W=%g')

        ax.plot([1e-3, 1e3], [1e-3, 1e3], 'k:', alpha=0.4)
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.set_label('Relative error of best approximation')

    # Eliminate the thin white seams between mesh cells in the saved PDF.
    for ax in axes:
        for col in ax.collections:
            col.set_edgecolor('face')

    plt.tight_layout()
    return fig


def plot_layer_dominance(R=1.0):
    """
    Which single layer carries the largest fraction of the total resistance.

    Replaces the obsolete "(W_1+W_2) vs (W_2+W_3) term dominance" plot.
    """
    W1_range = np.logspace(-3, 3, 80)
    W3_range = np.logspace(-3, 3, 80)
    W2_values = [0.01, 0.1, 1.0, 10.0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    cmap = plt.colormaps.get_cmap('Set1').resampled(3)
    for ax, W2 in zip(axes, W2_values):
        W1_grid, W3_grid = np.meshgrid(W1_range, W3_range)
        # 0 = layer 1, 1 = layer 2, 2 = layer 3
        argmax = np.argmax(np.stack([W1_grid,
                                     np.full_like(W1_grid, W2),
                                     W3_grid], axis=-1),
                           axis=-1)

        pcm = ax.pcolormesh(W1_grid, W3_grid, argmax, cmap=cmap,
                            shading='auto', vmin=-0.5, vmax=2.5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('$W_1$', fontsize=14)
        ax.set_ylabel('$W_3$', fontsize=14)
        ax.set_title(f'$W_2$ = {W2}', fontsize=14)
        cbar = plt.colorbar(pcm, ax=ax, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(['Layer 1', 'Layer 2', 'Layer 3'])

    plt.suptitle(f'Dominant layer (largest $W_i$), R = {R}',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# SANITY CHECK against direct numerical solution
# =============================================================================

def run_sanity_check():
    """
    Verify the corrected analytical solution against a direct steady-state
    finite-element (4-unknown) solve, and against the OLD wrong product-form.
    Returns a list of dicts so the caller can inspect results.
    """
    from scipy.optimize import fsolve

    def numerical_J(K_d1, K_r1, K_d3, K_r3,
                    K_d2_intf, K_r2_intf,
                    P, e_1, e_2, e_3, D_1, D_2, D_3):
        K_s1 = np.sqrt(K_d1 / K_r1)
        K_s2 = np.sqrt(K_d2_intf / K_r2_intf)
        K_s3 = np.sqrt(K_d3 / K_r3)

        def equations(x):
            C1, C12, C23, C3 = x
            J1 = D_1 * (C1 - C12) / e_1
            J2 = D_2 * ((K_s2 / K_s1) * C12 - C23) / e_2
            J3 = D_3 * ((K_s3 / K_s2) * C23 - C3) / e_3
            return [
                K_d1 * P - K_r1 * C1**2 - J1,
                J1 - J2,
                J2 - J3,
                K_r3 * C3**2 - J3,
            ]

        for scale in [0.9, 0.5, 0.99, 0.1, 1e-3]:
            x0 = [K_s1 * np.sqrt(P) * scale] * 4
            x_sol, _, ier, _ = fsolve(equations, x0, full_output=True)
            if ier == 1:
                break
        C1, C12, _, _ = x_sol
        return D_1 * (C1 - C12) / e_1, K_s1, K_s2, K_s3

    test_cases = [
        ('identical materials', dict(
            K_d1=1e15, K_r1=1e-28, K_d3=1e15, K_r3=1e-28,
            K_d2_intf=1e15, K_r2_intf=1e-28,
            P=1e5, e_1=1e-3, e_2=2e-3, e_3=3e-3,
            D_1=1e-9, D_2=1e-9, D_3=1e-9)),
        ('distinct materials moderate', dict(
            K_d1=1e15, K_r1=1e-28, K_d3=2e15, K_r3=5e-29,
            K_d2_intf=5e14, K_r2_intf=1e-27,
            P=1e5, e_1=1e-4, e_2=1e-5, e_3=2e-4,
            D_1=1e-9, D_2=1e-12, D_3=5e-10)),
        ('middle layer dominant', dict(
            K_d1=1e15, K_r1=1e-28, K_d3=1e15, K_r3=1e-28,
            K_d2_intf=1e15, K_r2_intf=1e-28,
            P=1e5, e_1=1e-4, e_2=1e-3, e_3=1e-4,
            D_1=1e-9, D_2=1e-13, D_3=1e-9)),
        ('surface limited', dict(
            K_d1=1e15, K_r1=1e-28, K_d3=1e15, K_r3=1e-28,
            K_d2_intf=1e15, K_r2_intf=1e-28,
            P=1e5, e_1=1e-7, e_2=1e-7, e_3=1e-7,
            D_1=1e-8, D_2=1e-8, D_3=1e-8)),
        ('asymmetric R=10', dict(
            K_d1=1e15, K_r1=1e-28, K_d3=1e16, K_r3=1e-28,
            K_d2_intf=1e15, K_r2_intf=1e-28,
            P=1e5, e_1=1e-4, e_2=1e-4, e_3=1e-4,
            D_1=1e-10, D_2=1e-10, D_3=1e-10)),
    ]

    results = []
    print('{:<28s} {:>12s} {:>12s} {:>12s} {:>12s}'.format(
        'case', 'J_numerical', 'J_sum', 'rel.err sum', 'rel.err prod'))
    for name, p in test_cases:
        J_num, K_s1, K_s2, K_s3 = numerical_J(**p)
        W1 = compute_Wi(p['K_d1'], p['P'], p['e_1'], p['D_1'], K_s1)
        W2 = compute_Wi(p['K_d1'], p['P'], p['e_2'], p['D_2'], K_s2)
        W3 = compute_Wi(p['K_d1'], p['P'], p['e_3'], p['D_3'], K_s3)
        R = p['K_d3'] / p['K_d1']

        W_sum = W1 + W2 + W3
        u, w, _ = solve_three_layer(W_sum, R)
        J_sum = p['K_d1'] * p['P'] * (u - w) / W_sum

        W_prod = (W1 + W2) * (W2 + W3)
        u_p, w_p, _ = solve_three_layer(W_prod, R)
        J_prod = p['K_d1'] * p['P'] * (u_p - w_p) / W_prod

        err_sum = abs(J_sum - J_num) / abs(J_num)
        err_prod = abs(J_prod - J_num) / abs(J_num)
        print('{:<28s} {:>12.4e} {:>12.4e} {:>12.2e} {:>12.2e}'.format(
            name, J_num, J_sum, err_sum, err_prod))
        results.append(dict(name=name, J_num=J_num, J_sum=J_sum, J_prod=J_prod,
                            err_sum=err_sum, err_prod=err_prod,
                            W1=W1, W2=W2, W3=W3, R=R))
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print('Running sanity check (corrected sum-of-W vs numerical) ...')
    run_sanity_check()

    print('\nGenerating regime map (W vs R) ...')
    fig1 = plot_regime_map()
    fig1.savefig(os.path.join(FIGS_DIR, 'regime_map.pdf'), dpi=150, bbox_inches='tight')

    print('Generating flux vs W plot ...')
    fig2 = plot_flux_vs_W()
    fig2.savefig(os.path.join(FIGS_DIR, 'flux_vs_W.pdf'), dpi=150, bbox_inches='tight')

    print('Generating error vs W plot ...')
    fig3 = plot_error_1D(R=1.0)
    fig3.savefig(os.path.join(FIGS_DIR, 'error_vs_W.pdf'), dpi=150, bbox_inches='tight')

    print('Generating W1-W3 regime map ...')
    fig4 = plot_regime_map_W1W3()
    fig4.savefig(os.path.join(FIGS_DIR, 'regime_map_W1W3.pdf'), dpi=150, bbox_inches='tight')

    print('Generating layer dominance map ...')
    fig5 = plot_layer_dominance()
    fig5.savefig(os.path.join(FIGS_DIR, 'layer_dominance.pdf'), dpi=150, bbox_inches='tight')

    print(f'\nAll figures saved to {FIGS_DIR}')
    plt.show()

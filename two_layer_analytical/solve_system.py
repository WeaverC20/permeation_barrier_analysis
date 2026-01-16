"""
Analytical solver for the system of equations:
    W(1-v²) = v - u = WRu²

This gives two equations:
    1. W(1-v²) = v - u
    2. v - u = WRu²

From (2): v = u + WRu²
Substituting into (1) yields a quartic in u:
    W²R²u⁴ + 2WRu³ + (1+R)u² - 1 = 0

Once u is found:
    v = u + WRu²
    J = v - u = WRu²
"""

from sympy import symbols, solve, simplify, sqrt, factor, collect, nsimplify


def get_analytical_solution():
    """
    Get the symbolic analytical solution for u, v, and J.
    """
    u, W, R = symbols('u W R', positive=True, real=True)

    # Quartic: W²R²u⁴ + 2WRu³ + (1+R)u² - 1 = 0
    equation = W**2 * R**2 * u**4 + 2*W*R*u**3 + (1 + R)*u**2 - 1

    # Solve symbolically
    u_solutions = solve(equation, u)

    # v = u + WRu²
    v_solutions = [sol + W*R*sol**2 for sol in u_solutions]

    # J = v - u = WRu²
    J_solutions = [W*R*sol**2 for sol in u_solutions]

    return u_solutions, v_solutions, J_solutions, W, R


def print_analytical_solution():
    """Print the reduced analytical solutions for u, v, and J."""
    u_sols, v_sols, J_sols, W, R = get_analytical_solution()

    print("="*70)
    print("ANALYTICAL SOLUTION for W(1-v²) = v - u = WRu²")
    print("="*70)

    print("\n" + "-"*70)
    print("GOVERNING EQUATIONS:")
    print("-"*70)
    print("  Quartic for u:  W²R²u⁴ + 2WRu³ + (1+R)u² - 1 = 0")
    print("  Then:           v = u + WRu²")
    print("                  J = v - u = WRu²")

    print("\n" + "-"*70)
    print("REDUCED ANALYTICAL SOLUTIONS:")
    print("-"*70)

    # Find the physical solution (first real positive one)
    for i, (u_sol, v_sol, J_sol) in enumerate(zip(u_sols, v_sols, J_sols)):
        u_simp = simplify(u_sol)
        v_simp = simplify(v_sol)
        J_simp = simplify(J_sol)

        print(f"\nSolution {i+1}:")
        print(f"  u = {u_simp}")
        print(f"  v = {v_simp}")
        print(f"  J = {J_simp}")

    print("\n" + "-"*70)
    print("LIMITING CASES (Simplified):")
    print("-"*70)

    # W -> 0 limit (surface-limited)
    print("\n  W → 0 (surface-limited):")
    print("    Equation reduces to: (1+R)u² - 1 = 0")
    print("    u = 1/√(1+R)")
    print("    v = 1/√(1+R)  (same as u)")
    print("    J = 0")

    # W -> ∞ limit (diffusion-limited)
    print("\n  W → ∞ (diffusion-limited):")
    print("    u → 0")
    print("    v → 1")
    print("    J → W(1-v²) → finite")

    # R -> 0 limit
    print("\n  R → 0:")
    print("    Equation reduces to: u² - 1 = 0")
    print("    u = 1, v = 1, J = 0")


def solve_system(W_val, R_val):
    """
    Solve W(1-v²) = v-u = WRu² for u, v, and J.

    Returns
    -------
    u, v, J : float
        Physical solutions
    """
    u = symbols('u')

    # Quartic: W²R²u⁴ + 2WRu³ + (1+R)u² - 1 = 0
    equation = (W_val**2 * R_val**2 * u**4
                + 2 * W_val * R_val * u**3
                + (1 + R_val) * u**2
                - 1)

    roots = solve(equation, u)

    # Find physical root (real and positive)
    physical_u = None
    for root in roots:
        root_eval = complex(root.evalf())
        if abs(root_eval.imag) < 1e-10 and root_eval.real > 0:
            physical_u = float(root_eval.real)
            break

    if physical_u is None:
        raise ValueError(f"No physical solution found for W={W_val}, R={R_val}")

    u_sol = physical_u
    v_sol = u_sol + W_val * R_val * u_sol**2
    J_sol = v_sol - u_sol  # = W*R*u²

    return u_sol, v_sol, J_sol


def verify_solution(W, R, u, v, J):
    """Verify that u, v, J satisfy all equations."""
    eq1 = W * (1 - v**2)      # W(1-v²)
    eq2 = v - u               # v - u
    eq3 = W * R * u**2        # WRu²

    print(f"  W(1-v²) = {eq1:.6f}")
    print(f"  v - u   = {eq2:.6f}")
    print(f"  WRu²    = {eq3:.6f}")
    print(f"  J       = {J:.6f}")
    print(f"  Max error: {max(abs(eq1-eq2), abs(eq2-eq3), abs(J-eq2)):.2e}")


if __name__ == "__main__":
    # Print analytical solution first
    print_analytical_solution()

    # Then numerical examples
    print("\n\n" + "="*70)
    print("NUMERICAL VERIFICATION")
    print("="*70)

    test_cases = [(0.1, 1.0), (1.0, 1.0), (10.0, 1.0), (1.0, 0.5), (1.0, 2.0)]

    for W, R in test_cases:
        print(f"\nW = {W}, R = {R}:")
        u, v, J = solve_system(W, R)
        print(f"  u = {u:.6f}")
        print(f"  v = {v:.6f}")
        print(f"  J = {J:.6f}")
        verify_solution(W, R, u, v, J)

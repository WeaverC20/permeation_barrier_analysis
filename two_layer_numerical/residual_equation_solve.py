# ---------- OPTION 3 (enforce u2(L) >= 0 by reparameterization) ----------
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# --- Parameters (numerical) ---
D1 = 6e-11           # inconel
D2 = 1.54e-11        # 316L steel
x0 = 0.0
xint = 0.5
L = 1.0
P = 1e5

Kd1 = 3e13
Kr1 = 2.68e-30
Kd2 = 8e11
Kr2 = 2.4e-33

S1 = float((Kd1 / Kr1) ** 0.5)
S2 = float((Kd2 / Kr2) ** 0.5)
gamma = S2 / S1

# --- Natural scales ---
C_scale = float((Kd1 * P / Kr1) ** 0.5)
J_scale = float(Kd1 * P)
A1_scale = J_scale / D1
A2_scale = J_scale / D2
alpha1 = A1_scale / C_scale
alpha2 = A2_scale / C_scale

def unpack_from_hat(vhat):
    """vhat = [a1_hat, a2_hat, b1_hat, q_hat]; with u2L_hat = q_hat**2."""
    a1_hat, a2_hat, b1_hat, q_hat = vhat
    u2L_hat = q_hat**2                      # >= 0 by construction
    b2_hat  = u2L_hat - alpha2*a2_hat*L     # so that u2(L) = a2*L + b2 >= 0
    a1 = a1_hat * A1_scale
    a2 = a2_hat * A2_scale
    b1 = b1_hat * C_scale
    b2 = b2_hat * C_scale
    return a1, a2, b1, b2

def residuals_hat(vhat):
    a1, a2, b1, b2 = unpack_from_hat(vhat)
    # interface values
    u1_int = a1 * xint + b1
    u2_int = a2 * xint + b2
    # right boundary value
    u2L = a2 * L + b2  # guaranteed >= 0

    r0 = (u1_int / S1) - (u2_int / S2)                   # interface equilibrium
    r1 = (D1 * a1 - D2 * a2) / J_scale                   # flux continuity (dimensionless)
    r2 = (-D1 * a1 - (Kd1 * P - Kr1 * b1**2)) / J_scale  # left surface
    r3 = (-D2 * a2 + Kr2 * (u2L**2)) / J_scale           # right surface (uses u2L >= 0)
    return np.array([r0, r1, r2, r3], dtype=float)

# Seeds in hat variables: [a1_hat, a2_hat, b1_hat, q_hat]
# Start with tiny slopes, b1_hat ≈ 1, and q_hat ≈ sqrt(u2L_hat) ~ O(1)
seeds = [
    np.array([0.0, 0.0, 1.0, 1.0]),
    np.array([0.0, 0.0, 1.05, 1.0]),
    np.array([0.0, 0.0, 0.95, 1.0]),
    np.array([1e-3, 1e-3, 1.0, 1.0]),
]

solution = None
last_msg = None
for x0_hat in seeds:
    try:
        res = least_squares(
            residuals_hat,
            x0_hat,
            method="trf",
            jac="2-point",
            loss="soft_l1",
            x_scale=1.0,
            bounds=([-np.inf, -np.inf, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]),  # b1_hat>=0, q_hat>=0
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
            max_nfev=20000
        )
        if res.success:
            solution = res
            break
        last_msg = res.message
    except Exception as e:
        last_msg = str(e)

if solution is None:
    raise RuntimeError(f"least_squares failed to converge with tried seeds. Last message: {last_msg}")

a1_hat, a2_hat, b1_hat, q_hat = solution.x
a1, a2, b1, b2 = unpack_from_hat(solution.x)
u2L = a2*L + b2

print("Converged:", solution.success, "| nfev:", solution.nfev, "| ||r||2:", np.linalg.norm(solution.fun))
print(f"a1={a1:.6e}, a2={a2:.6e}, b1={b1:.6e}, b2={b2:.6e}, u2(L)={u2L:.6e}  (forced >= 0)")

# Plot
x1_vals = np.linspace(x0, xint, 400)
x2_vals = np.linspace(xint, L, 400)
u1_vals = a1 * x1_vals + b1
u2_vals = a2 * x2_vals + b2

print(f"C_left  = u1(x0) = {u1_vals[0]:.6e}")
print(f"C_right = u2(L)  = {u2_vals[-1]:.6e}")

plt.figure(figsize=(8,5))
plt.plot(x1_vals, u1_vals, label="u1(x)")
plt.plot(x2_vals, u2_vals, label="u2(x)")
plt.axvline(xint, linestyle="--", label="x_int")
plt.xlabel("x"); plt.ylabel("u(x)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()
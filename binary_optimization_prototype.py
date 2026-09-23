import numpy as np
from scipy.optimize import linprog, milp, minimize, Bounds, LinearConstraint
from time import time
import pandas as pd

# ============================================================
# Problem generation
# ============================================================
N = 100
M = 40
np.random.seed(123)
c = np.random.randint(1, 10, size=N).astype(float)       # maximize c @ x
A = np.random.randint(0, 10, size=(M, N)).astype(float)  # constraint matrix
lb_constr = np.zeros(M)
ub_constr = 100.0 * np.ones(M)

c_min = -c  # for minimization (scipy minimizes)

# Constraints: lb <= Ax <= ub  →  A_ub @ x <= b_ub
A_ub = np.vstack([A, -A])
b_ub = np.concatenate([ub_constr, -lb_constr])

# ============================================================
# Algorithm parameters
# ============================================================
EPS = 0.1        # tolerance for rounding (fix if |x - round(x)| < eps)
EPS_FINE = 0.01  # fine tolerance for conservative fixing
ALPHA = 0.2      # batch fraction
BETA = 1.5       # penalty growth factor
RHO_MAX = 1e4    # penalty ceiling
K_THRESH = 30    # threshold for switching to exact MILP
NO_PROGRESS_LIMIT = 5
MAX_LIN_ITER = 50
MAX_LP_ITER = 100
NLP_GAMMAS = [1e2, 1e3, 1e4, 1e5, 1e6]

# ============================================================
# Helper: solve LP on free vars J with fixed vars I
# ============================================================
def solve_lp(I_set, J_set, x_fixed, mod_c=None):
    """LP: min c_min[J] @ x_J  s.t.  A_ub[:,J] @ x_J <= b_ub - A_ub[:,I] @ x_I,  0<=x<=1.
    
    mod_c: optional array added to objective coefficients (for LP-linearization).
    Returns (x_full, obj_val, |reduced_costs|) or (None, None, None) if infeasible.
    """
    J = np.array(sorted(J_set), dtype=int)
    I = np.array(sorted(I_set), dtype=int) if len(I_set) > 0 else np.array([], dtype=int)
    
    if len(J) == 0:
        return x_fixed.copy(), c_min @ x_fixed, np.array([])
    
    obj = c_min[J].copy()
    if mod_c is not None:
        obj = obj + mod_c
    
    A_J = A_ub[:, J]
    b = b_ub.copy()
    if len(I) > 0:
        b = b - A_ub[:, I] @ x_fixed[I]
    
    res = linprog(c=obj, A_ub=A_J, b_ub=b, bounds=(0, 1), method='highs')
    if not res.success:
        return None, None, None
    
    x_full = x_fixed.copy()
    x_full[J] = res.x
    
    # Reduced costs (absolute value = confidence measure)
    rc = np.zeros(len(J))
    try:
        if hasattr(res, 'lower') and res.lower is not None:
            rc += np.abs(res.lower.marginals)
        if hasattr(res, 'upper') and res.upper is not None:
            rc += np.abs(res.upper.marginals)
    except Exception:
        rc = np.abs(res.x - 0.5)
    
    return x_full, res.fun, rc


# ============================================================
# Helper: solve exact MILP on free vars J
# ============================================================
def solve_milp(I_set, J_set, x_fixed):
    """MILP: min c_min[J] @ x_J  s.t.  A_ub[:,J] @ x_J <= b_ub - A_ub[:,I] @ x_I,  x in {0,1}.
    Returns (x_full, obj_val) or (None, None) if infeasible.
    """
    J = np.array(sorted(J_set), dtype=int)
    I = np.array(sorted(I_set), dtype=int) if len(I_set) > 0 else np.array([], dtype=int)
    
    if len(J) == 0:
        return x_fixed.copy(), c_min @ x_fixed
    
    obj = c_min[J]
    A_J = A_ub[:, J]
    b = b_ub.copy()
    if len(I) > 0:
        b = b - A_ub[:, I] @ x_fixed[I]
    
    res = milp(c=obj,
               bounds=Bounds(lb=np.zeros(len(J)), ub=np.ones(len(J))),
               constraints=LinearConstraint(A=A_J, ub=b),
               integrality=[1] * len(J))
    if not res.success:
        return None, None
    
    x_full = x_fixed.copy()
    x_full[J] = res.x
    return x_full, res.fun


# ============================================================
# Helper: try to fix a batch of variables
# ============================================================
def try_fix(I_set, J_set, x_fixed, batch, x_vals):
    """Try fixing batch of vars to round(x_vals[batch]). Check LP feasibility."""
    new_x = x_fixed.copy()
    for i in batch:
        new_x[i] = round(x_vals[i])
    new_I = I_set | set(int(i) for i in batch)
    new_J = J_set - set(int(i) for i in batch)
    x, _, _ = solve_lp(new_I, new_J, new_x)
    return (x is not None), new_I, new_J, new_x


# ============================================================
# Helper: batch fixing with binary-search backtracking
# ============================================================
def fix_batch(I_set, J_set, x_fixed, candidates, x_vals, alpha):
    """Fix candidates in batches. On failure, halve batch size (binary search).
    Returns (new_I, new_J, new_x_fixed, fix_history).
    """
    fix_history = []
    if len(candidates) == 0:
        return I_set, J_set, x_fixed, fix_history
    
    # Sort by distance from 0.5 (most confident = closest to 0/1 first)
    dist = np.abs(x_vals[candidates] - 0.5)
    order = np.argsort(-dist)
    cand_sorted = candidates[order]
    
    batch_size = min(int(np.ceil(alpha * len(cand_sorted))), len(cand_sorted))
    
    while batch_size > 0:
        batch = cand_sorted[:batch_size]
        ok, new_I, new_J, new_x = try_fix(I_set, J_set, x_fixed, batch, x_vals)
        if ok:
            fix_history.append(list(int(i) for i in batch))
            return new_I, new_J, new_x, fix_history
        batch_size = batch_size // 2
    
    return I_set, J_set, x_fixed, fix_history


# ============================================================
# Phase 1: LP fix-and-propagate
# ============================================================
def phase1_lp(I_set, J_set, x_fixed, eps=EPS, alpha=ALPHA, max_iter=MAX_LP_ITER):
    """Repeatedly solve LP, fix near-binary variables in batches with backtracking.
    Returns (ok, I, J, x_fixed, fix_history, lp_obj).
    """
    fix_history_all = []
    lp_obj = None
    
    for _ in range(max_iter):
        x, lp_obj, rc = solve_lp(I_set, J_set, x_fixed)
        if x is None:
            return False, I_set, J_set, x_fixed, fix_history_all, lp_obj
        
        J_arr = np.array(sorted(J_set), dtype=int)
        if len(J_arr) == 0:
            break
        
        mask = np.abs(x[J_arr] - np.round(x[J_arr])) < eps
        candidates = J_arr[mask]
        
        if len(candidates) == 0:
            break
        
        new_I, new_J, new_x, hist = fix_batch(I_set, J_set, x_fixed, candidates, x, alpha)
        if len(hist) == 0:
            break
        
        I_set, J_set, x_fixed = new_I, new_J, new_x
        fix_history_all.extend(hist)
    
    return True, I_set, J_set, x_fixed, fix_history_all, lp_obj


# ============================================================
# Phase 2a: LP-Linearization (homotopy with linearized concave penalty)
# ============================================================
def phase2_lp_lin(I_set, J_set, x_fixed, eps=EPS, alpha=ALPHA, beta=BETA,
                  rho_max=RHO_MAX, max_iter=MAX_LIN_ITER,
                  no_progress_limit=NO_PROGRESS_LIMIT, k_thresh=K_THRESH):
    """LP-linearization phase.
    
    Replaces NLP min(c + rho*sum(x_i*(1-x_i))) with a sequence of LP:
      min (c_min[J] + rho*(1 - 2*x_bar)) @ x_J   [linearized objective]
    
    rho grows geometrically; variables deeply fractional at 0.5 get faster growth.
    Near-binary variables are fixed in batches with backtracking.
    """
    fix_history_all = []
    
    J_arr = np.array(sorted(J_set), dtype=int)
    if len(J_arr) == 0:
        return True, I_set, J_set, x_fixed, fix_history_all
    
    # Initialize rho: small, proportional to |c_i|
    rho = np.abs(c[J_arr]) * 0.0025
    no_progress = 0
    
    for iteration in range(max_iter):
        J_arr = np.array(sorted(J_set), dtype=int)
        if len(J_arr) == 0:
            break
        
        # Current LP solution (for linearization point)
        x_base, _, _ = solve_lp(I_set, J_set, x_fixed)
        if x_base is None:
            return False, I_set, J_set, x_fixed, fix_history_all
        
        x_bar = x_base[J_arr]
        
        # Linearized penalty: rho_i * (1 - 2*x_bar_i) added to c_min
        # If x_bar < 0.5 → positive addition → pushes x_i toward 0
        # If x_bar > 0.5 → negative addition → pushes x_i toward 1
        mod_c = rho * (1 - 2 * x_bar)
        
        x_new, _, rc_new = solve_lp(I_set, J_set, x_fixed, mod_c=mod_c)
        if x_new is None:
            return False, I_set, J_set, x_fixed, fix_history_all
        
        x_new_J = x_new[J_arr]
        
        # Fix near-binary
        mask = np.abs(x_new_J - np.round(x_new_J)) < eps
        candidates = J_arr[mask]
        
        if len(candidates) > 0:
            new_I, new_J, new_x, hist = fix_batch(I_set, J_set, x_fixed, candidates, x_new, alpha)
            if len(hist) > 0:
                I_set, J_set, x_fixed = new_I, new_J, new_x
                fix_history_all.extend(hist)
                no_progress = 0
                # Update rho for remaining J
                new_J_arr = np.array(sorted(J_set), dtype=int)
                old_rho = {int(j): rho[idx] for idx, j in enumerate(J_arr)}
                rho = np.array([old_rho.get(int(j), np.abs(c[int(j)]) * 0.0025) for j in new_J_arr])
            else:
                no_progress += 1
        else:
            no_progress += 1
        
        # Increase rho (faster for deeply fractional variables)
        for idx in range(len(J_arr)):
            if idx < len(x_new_J) and abs(x_new_J[idx] - 0.5) < 0.1:
                rho[idx] = min(rho[idx] * beta**2, rho_max)
            else:
                rho[idx] = min(rho[idx] * beta, rho_max)
        
        # Termination checks
        if len(J_set) <= k_thresh:
            break
        if no_progress >= no_progress_limit:
            break
        if np.all(rho >= rho_max):
            break
    
    return True, I_set, J_set, x_fixed, fix_history_all


# ============================================================
# Phase 2b: NLP Penalty (for comparison — concave minimization)
# ============================================================
def phase2_nlp(I_set, J_set, x_fixed, eps=EPS, gammas=None):
    """NLP penalty phase using scipy.optimize.minimize (SLSQP).
    
    Solves: min c_min[J]@x + gamma * sum(x_i*(1-x_i))  s.t.  linear constraints, 0<=x<=1.
    NOTE: objective is concave → SLSQP finds local (not global) minimum.
    """
    if gammas is None:
        gammas = NLP_GAMMAS
    
    fix_history_all = []
    
    # Start from LP solution
    x_start, _, _ = solve_lp(I_set, J_set, x_fixed)
    if x_start is None:
        return False, I_set, J_set, x_fixed, fix_history_all
    
    no_progress = 0
    
    for gamma in gammas:
        J_arr = np.array(sorted(J_set), dtype=int)
        if len(J_arr) == 0:
            break
        
        x0 = x_start[J_arr].copy()
        
        def obj_nlp(x_J, g=gamma, j=J_arr):
            return c_min[j] @ x_J + g * np.sum(x_J * (1 - x_J))
        
        def grad_nlp(x_J, g=gamma, j=J_arr):
            return c_min[j] + g * (1 - 2 * x_J)
        
        I_arr = np.array(sorted(I_set), dtype=int) if len(I_set) > 0 else np.array([], dtype=int)
        b = b_ub.copy()
        if len(I_arr) > 0:
            b = b - A_ub[:, I_arr] @ x_fixed[I_arr]
        A_J = A_ub[:, J_arr]
        
        lc = LinearConstraint(A=A_J, ub=b)
        bds = Bounds(lb=np.zeros(len(J_arr)), ub=np.ones(len(J_arr)))
        
        res = minimize(fun=obj_nlp, x0=x0, jac=grad_nlp,
                       bounds=bds, constraints=[lc],
                       method='SLSQP', options={'maxiter': 200, 'ftol': 1e-8})
        
        x_start = x_fixed.copy()
        x_start[J_arr] = res.x
        
        # Fix near-binary
        mask = np.abs(res.x - np.round(res.x)) < eps
        candidates = J_arr[mask]
        
        if len(candidates) > 0:
            new_I, new_J, new_x, hist = fix_batch(I_set, J_set, x_fixed, candidates, x_start, 0.3)
            if len(hist) > 0:
                I_set, J_set, x_fixed = new_I, new_J, new_x
                fix_history_all.extend(hist)
                no_progress = 0
            else:
                no_progress += 1
        else:
            no_progress += 1
        
        if no_progress >= 3:
            break
    
    return True, I_set, J_set, x_fixed, fix_history_all


# ============================================================
# Phase 3: Exact MILP on remaining variables
# ============================================================
def phase3_exact(I_set, J_set, x_fixed):
    """Solve exact MILP on free variables J. Returns (x_full, obj_val, |J|) or (None, None, |J|)."""
    x_full, obj = solve_milp(I_set, J_set, x_fixed)
    if x_full is None:
        return None, None, len(J_set)
    return x_full, obj, len(J_set)


# ============================================================
# Main: run all algorithms and compare
# ============================================================
def run_all():
    results = []
    
    # --- Reference: LP relaxation ---
    t0 = time()
    res_lp = linprog(c=c_min, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), method='highs')
    t_lp = time() - t0
    x_lp = res_lp.x
    val_lp = c @ x_lp
    
    # --- Reference: Direct MILP ---
    t0 = time()
    res_milp = milp(c=c_min, bounds=Bounds(0, 1),
                    constraints=LinearConstraint(A=A_ub, ub=b_ub),
                    integrality=[1]*N)
    t_milp = time() - t0
    x_milp = res_milp.x
    val_milp = c @ x_milp
    
    results.append({'Algorithm': 'LP relaxation', 'Value': val_lp, 'Time': t_lp,
                    'Gap%': (val_milp - val_lp)/abs(val_milp)*100, '|J|': N, 'OK': True})
    results.append({'Algorithm': 'Direct MILP (exact)', 'Value': val_milp, 'Time': t_milp,
                    'Gap%': 0.0, '|J|': 0, 'OK': True})
    
    # --- Algorithm A: LP + LP-Lin + Exact ---
    I, J, x = set(), set(range(N)), np.zeros(N)
    t0 = time()
    _, I, J, x, _, _ = phase1_lp(I, J, x, eps=EPS)
    n_p1 = len(J)
    _, I, J, x, _ = phase2_lp_lin(I, J, x, eps=EPS)
    n_p2 = len(J)
    if n_p2 > 0:
        x_f, _, _ = phase3_exact(I, J, x)
    else:
        x_f = x
    t_a = time() - t0
    val_a = c @ x_f if x_f is not None else None
    gap_a = (val_milp - val_a)/abs(val_milp)*100 if val_a is not None else None
    results.append({'Algorithm': 'A: LP+LP-Lin+Exact', 'Value': val_a, 'Time': t_a,
                    'Gap%': gap_a, '|J|': n_p2, 'OK': x_f is not None})
    
    # --- Algorithm A': LP + LP-Lin + Exact (eps=0.01) ---
    I, J, x = set(), set(range(N)), np.zeros(N)
    t0 = time()
    _, I, J, x, _, _ = phase1_lp(I, J, x, eps=EPS_FINE)
    _, I, J, x, _ = phase2_lp_lin(I, J, x, eps=EPS_FINE)
    if len(J) > 0:
        x_f, _, _ = phase3_exact(I, J, x)
    else:
        x_f = x
    t_a2 = time() - t0
    val_a2 = c @ x_f if x_f is not None else None
    gap_a2 = (val_milp - val_a2)/abs(val_milp)*100 if val_a2 is not None else None
    results.append({'Algorithm': "A': LP+LP-Lin+Exact (eps=0.01)", 'Value': val_a2, 'Time': t_a2,
                    'Gap%': gap_a2, '|J|': len(J), 'OK': x_f is not None})
    
    # --- Algorithm B: LP + NLP + Exact ---
    I, J, x = set(), set(range(N)), np.zeros(N)
    t0 = time()
    _, I, J, x, _, _ = phase1_lp(I, J, x, eps=EPS)
    _, I, J, x, _ = phase2_nlp(I, J, x, eps=EPS)
    if len(J) > 0:
        x_f, _, _ = phase3_exact(I, J, x)
    else:
        x_f = x
    t_b = time() - t0
    val_b = c @ x_f if x_f is not None else None
    gap_b = (val_milp - val_b)/abs(val_milp)*100 if val_b is not None else None
    results.append({'Algorithm': 'B: LP+NLP+Exact', 'Value': val_b, 'Time': t_b,
                    'Gap%': gap_b, '|J|': len(J), 'OK': x_f is not None})
    
    # --- Algorithm C: LP + Exact (no Phase 2) ---
    I, J, x = set(), set(range(N)), np.zeros(N)
    t0 = time()
    _, I, J, x, _, _ = phase1_lp(I, J, x, eps=EPS)
    if len(J) > 0:
        x_f, _, _ = phase3_exact(I, J, x)
    else:
        x_f = x
    t_c = time() - t0
    val_c = c @ x_f if x_f is not None else None
    gap_c = (val_milp - val_c)/abs(val_milp)*100 if val_c is not None else None
    results.append({'Algorithm': 'C: LP+Exact', 'Value': val_c, 'Time': t_c,
                    'Gap%': gap_c, '|J|': len(J), 'OK': x_f is not None})
    
    # --- Print results ---
    df = pd.DataFrame(results)
    print("="*85)
    print(f"  PROBLEM: N={N}, M={M},  maximize c@x,  0<=Ax<=100,  x in {{0,1}}")
    print("="*85)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if abs(x)<100 else f"{x:.1f}"))
    print("="*85)
    
    print(f"\n  Phase breakdown (Algorithm A, eps={EPS}):")
    print(f"    Phase 1 (LP):          |J| {N} → {n_p1}")
    print(f"    Phase 2a (LP-Lin):     |J| {n_p1} → {n_p2}")
    print(f"    Phase 3 (Exact MILP):  |J| {n_p2} → 0")
    print(f"    Total time: {t_a:.4f}s  vs  MILP {t_milp:.4f}s  ({t_milp/t_a:.1f}x speedup)")
    
    print(f"\n  Phase breakdown (Algorithm A', eps={EPS_FINE}):")
    print(f"    Total time: {t_a2:.4f}s  vs  MILP {t_milp:.4f}s  ({t_milp/t_a2:.1f}x speedup)")
    if gap_a2 is not None:
        print(f"    Gap: {gap_a2:.2f}%  (exact match with MILP!)" if gap_a2 == 0 else f"    Gap: {gap_a2:.2f}%")
    
    return df

if __name__ == '__main__':
    df = run_all()

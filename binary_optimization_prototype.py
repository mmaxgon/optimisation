import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from time import time
import pandas as pd


# ============================================================
# Генерация задачи
# ============================================================
N = 100
M = 40
np.random.seed(123)
c = np.random.randint(1, 10, size=N).astype(float)       # максимизируем c @ x
A = np.random.randint(0, 10, size=(M, N)).astype(float)
lb_constr = np.zeros(M)
ub_constr = 100.0 * np.ones(M)

c_min = -c  # scipy минимизирует

# Ограничения: lb <= Ax <= ub  →  A_ub @ x <= b_ub
A_ub = np.vstack([A, -A])
b_ub = np.concatenate([ub_constr, -lb_constr])


# ============================================================
# Параметры алгоритма (см. раздел "Алгоритм")
# ============================================================
EPS = 1e-6            # порог целочисленности ε
RHO_0 = 1e-3          # начальный штрафной коэффициент ρ
GAMMA = 0.1           # параметр гарантии ненулевого штрафа γ
BETA = 1.5            # коэффициент увеличения штрафа β
RHO_MAX = 1e4         # потолок штрафа ρ_max
K_THRESH = 30         # порог |J| для перехода на Шаг 3
MAX_ITER = 200        # предохранитель от зацикливания


# ============================================================
# Вспомогательные функции
# ============================================================
def solve_lp(I, J, x_fixed, mod_c=None):
    """LP: min c_min[J] @ x_J + mod_c @ x_J
    при A_ub[:,J] @ x_J <= b_ub - A_ub[:,I] @ x_I,  0 <= x <= 1.
    Возвращает (x_full, obj) или (None, None) при недопустимости.
    """
    J_arr = np.array(sorted(J), dtype=int)
    I_arr = np.array(sorted(I), dtype=int) if I else np.array([], dtype=int)

    # --- ИСПРАВЛЕНИЕ: проверка ограничений при J = ∅ ---
    if len(J_arr) == 0:
        # Проверяем, что x_fixed удовлетворяет ограничениям
        violation = A_ub @ x_fixed - b_ub
        if np.all(violation <= 1e-8):
            return x_fixed.copy(), float(c_min @ x_fixed)
        else:
            return None, None

    obj = c_min[J_arr].copy()
    if mod_c is not None:
        obj = obj + mod_c

    b = b_ub.copy()
    if len(I_arr) > 0:
        b = b - A_ub[:, I_arr] @ x_fixed[I_arr]

    res = linprog(c=obj, A_ub=A_ub[:, J_arr], b_ub=b, bounds=(0, 1), method='highs')
    if not res.success:
        return None, None

    x_full = x_fixed.copy()
    x_full[J_arr] = res.x
    return x_full, float(res.fun)


def solve_milp(I, J, x_fixed):
    """MILP на свободных переменных J. Возвращает (x_full, obj) или (None, None)."""
    J_arr = np.array(sorted(J), dtype=int)
    I_arr = np.array(sorted(I), dtype=int) if I else np.array([], dtype=int)

    if len(J_arr) == 0:
        violation = A_ub @ x_fixed - b_ub
        if np.all(violation <= 1e-8):
            return x_fixed.copy(), float(c_min @ x_fixed)
        else:
            return None, None

    b = b_ub.copy()
    if len(I_arr) > 0:
        b = b - A_ub[:, I_arr] @ x_fixed[I_arr]

    res = milp(c=c_min[J_arr],
               bounds=Bounds(lb=np.zeros(len(J_arr)), ub=np.ones(len(J_arr))),
               constraints=LinearConstraint(A=A_ub[:, J_arr], ub=b),
               integrality=[1] * len(J_arr))
    if not res.success:
        return None, None

    x_full = x_fixed.copy()
    x_full[J_arr] = res.x
    return x_full, float(res.fun)


def is_binary(x, eps=EPS):
    """Почти бинарная компонента: |x - round(x)| < ε."""
    return abs(x - round(x)) < eps


def split_I_half(I_cand, x_vals):
    """Делит I_cand пополам, оставляя половину с минимальными
    отклонениями от бинарности. Возвращает (keep, moved).
    """
    deviations = [(i, abs(x_vals[i] - round(x_vals[i]))) for i in I_cand]
    deviations.sort(key=lambda t: t[1])
    half = max(1, len(deviations) // 2)
    keep = set(i for i, _ in deviations[:half])
    moved = set(i for i, _ in deviations[half:])
    return keep, moved


def try_fix(I, J, x_fixed, I_cand, x_vals):
    """Пытается зафиксировать кандидатов I_cand, округляя их.
    При недопустимости делит I_cand пополам (бисекция множества).
    Возвращает (ok, I, J, x_fixed).
    """
    I_cand = set(I_cand)
    if not I_cand:
        return False, I, J, x_fixed

    # Округляем кандидатов
    x_new = x_fixed.copy()
    for i in I_cand:
        x_new[i] = round(x_vals[i])

    new_I = I | I_cand
    new_J = J - I_cand
    x_test, _ = solve_lp(new_I, new_J, x_new)

    if x_test is not None:
        return True, new_I, new_J, x_new

    # --- ИСПРАВЛЕНИЕ BUG 1: проверка |I_cand| <= 1 перед рекурсией ---
    if len(I_cand) <= 1:
        return False, I, J, x_fixed

    # Недопустимо — бисекция множества I_cand
    I_keep, _ = split_I_half(I_cand, x_vals)
    return try_fix(I, J, x_fixed, I_keep, x_vals) if I_keep else (False, I, J, x_fixed)


# ============================================================
# Шаг 0 и Шаг 1: округление почти бинарных переменных
# ============================================================
def rounding_step(I, J, x_fixed, came_from, eps=EPS, max_iter=MAX_ITER):
    """Шаг 0 (came_from='start') или Шаг 1 (came_from='step2').
    Возвращает (status, I, J, x_fixed, next_action).
    """
    if not J:
        return 'return_true', I, J, x_fixed, 'done'

    if len(J) <= K_THRESH:
        return 'goto_step3', I, J, x_fixed, 'step3'

    # --- ИСПРАВЛЕНИЕ BUG 3: при переходе из Шага 2 сначала проверяем
    #     входящее решение (LP-3), а не решаем plain LP заново ---
    if came_from == 'step2':
        I_cand_init = set(i for i in J if is_binary(x_fixed[i], eps))
        if I_cand_init:
            ok, new_I, new_J, new_x = try_fix(I, J, x_fixed, I_cand_init, x_fixed)
            if ok:
                I, J, x_fixed = new_I, new_J, new_x
                if not J:
                    return 'return_true', I, J, x_fixed, 'done'
                if len(J) <= K_THRESH:
                    return 'goto_step3', I, J, x_fixed, 'step3'
                # Успешно зафиксировали из LP-3 — продолжаем цикл с plain LP
            # Если не удалось — продолжаем с plain LP ниже

    for _ in range(max_iter):
        x, _ = solve_lp(I, J, x_fixed)
        if x is None:
            return ('fail' if came_from == 'step2' else 'goto_step2'), I, J, x_fixed, \
                   ('done' if came_from == 'step2' else 'step2')

        I_cand = set(i for i in J if is_binary(x[i], eps))

        if not I_cand:
            if came_from == 'step2':
                return 'fail', I, J, x_fixed, 'done'
            return 'goto_step2', I, J, x_fixed, 'step2'

        ok, new_I, new_J, new_x = try_fix(I, J, x_fixed, I_cand, x)

        if not ok:
            if came_from == 'step2':
                return 'fail', I, J, x_fixed, 'done'
            return 'goto_step2', I, J, x_fixed, 'step2'

        I, J, x_fixed = new_I, new_J, new_x

        if not J:
            return 'return_true', I, J, x_fixed, 'done'
        if len(J) <= K_THRESH:
            return 'goto_step3', I, J, x_fixed, 'step3'

    if came_from == 'step2':
        return 'fail', I, J, x_fixed, 'done'
    return 'goto_step2', I, J, x_fixed, 'step2'


# ============================================================
# Шаг 2: линеаризованный штраф
# ============================================================
def penalty_step(I, J, x_fixed, rho, eps=EPS, max_iter=MAX_ITER):
    """Шаг 2. Решает (LP-3) с линеаризованным штрафом.
    Возвращает (status, I, J, x_fixed, rho, next_action).
    """
    if not J:
        return 'return_true', I, J, x_fixed, rho, 'done'

    J_arr = np.array(sorted(J), dtype=int)

    # --- ИСПРАВЛЕНИЕ BUG 2: начальная точка линеаризации ---
    # Стартуем из входящего решения (plain LP)
    x_current, _ = solve_lp(I, J, x_fixed)
    if x_current is None:
        return 'fail', I, J, x_fixed, rho, 'done'

    for _ in range(max_iter):
        x_bar = x_current[J_arr]   # точка линеаризации — обновляется!

        # Индивидуальные штрафные множители:
        #   ρ_i = ρ · ((1 − 2·x̄_i)² + γ)
        rho_i = rho * ((1 - 2 * x_bar) ** 2 + GAMMA)

        # Линеаризованный штраф: добавляем ρ_i · (1 − 2·x̄_i) к c_min
        mod_c = rho_i * (1 - 2 * x_bar)

        # Решаем (LP-3) с модифицированной целевой функцией
        x_new, _ = solve_lp(I, J, x_fixed, mod_c=mod_c)
        if x_new is None:
            return 'fail', I, J, x_fixed, rho, 'done'

        # Кандидаты на округление
        I_cand = set(i for i in J if is_binary(x_new[i], eps))

        if I_cand:
            # Сбрасываем ρ в начальное значение и возвращаемся на Шаг 1
            return 'goto_step1', I, J, x_new, RHO_0, 'step1'

        # --- ИСПРАВЛЕНИЕ BUG 2: обновляем точку линеаризации ---
        x_current = x_new

        # Кандидатов нет — увеличиваем штраф
        rho = rho * BETA
        if rho > RHO_MAX:
            return 'fail', I, J, x_fixed, rho, 'done'

    return 'fail', I, J, x_fixed, rho, 'done'


# ============================================================
# Шаг 3: точное решение MILP
# ============================================================
def exact_step(I, J, x_fixed, k_thresh=K_THRESH):
    """Шаг 3. Решает MILP на J.
    При несовместности откатывает последний батч и увеличивает K.
    Возвращает (x_full, obj) или (None, None).
    """
    x_full, obj = solve_milp(I, J, x_fixed)
    if x_full is not None:
        return x_full, obj

    # MILP несовместна — увеличиваем K и пробуем с большим |J|
    # (упрощённая версия отката: просто возвращаем None,
    #  вызывающий код может повторить с большим K)
    return None, None


# ============================================================
# Главный цикл алгоритма
# ============================================================
def run_algorithm(k_thresh=K_THRESH, eps=EPS):
    """Основной алгоритм: Шаг 0 → Шаг 1 ↔ Шаг 2 → Шаг 3."""
    I, J = set(), set(range(N))
    x_fixed = np.zeros(N)

    # Шаг 0: начальное округление (came_from='start')
    status, I, J, x_fixed, action = rounding_step(I, J, x_fixed, came_from='start', eps=eps)
    if status == 'return_true':
        return x_fixed, c @ x_fixed
    if status == 'fail':
        return None, None

    rho = RHO_0

    # Основной цикл: Шаг 1 ↔ Шаг 2
    for _ in range(MAX_ITER):
        if action == 'step3':
            break
        if action == 'step2':
            status, I, J, x_fixed, rho, action = penalty_step(I, J, x_fixed, rho, eps=eps)
            if status == 'return_true':
                return x_fixed, c @ x_fixed
            if status == 'fail':
                return None, None
            continue
        if action == 'step1':
            status, I, J, x_fixed, action = rounding_step(I, J, x_fixed, came_from='step2', eps=eps)
            if status == 'return_true':
                return x_fixed, c @ x_fixed
            if status == 'fail':
                return None, None
            continue
        if action == 'done':
            break

    # Шаг 3: точное MILP
    x_full, obj = exact_step(I, J, x_fixed)
    if x_full is None:
        return None, None
    return x_full, c @ x_full


# ============================================================
# Сравнение с эталонами
# ============================================================
def run_all():
    results = []

    # Эталон 1: LP-релаксация
    t0 = time()
    res_lp = linprog(c=c_min, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), method='highs')
    t_lp = time() - t0
    val_lp = c @ res_lp.x

    # Эталон 2: прямое MILP
    t0 = time()
    res_milp = milp(c=c_min, bounds=Bounds(0, 1),
                    constraints=LinearConstraint(A=A_ub, ub=b_ub),
                    integrality=[1] * N)
    t_milp = time() - t0
    val_milp = c @ res_milp.x

    results.append({'Алгоритм': 'LP-релаксация', 'Значение': val_lp, 'Время': t_lp,
                    'Разрыв %': (val_milp - val_lp) / abs(val_milp) * 100})
    results.append({'Алгоритм': 'Прямое MILP', 'Значение': val_milp, 'Время': t_milp,
                    'Разрыв %': 0.0})

    # Наш алгоритм
    t0 = time()
    x_alg, val_alg = run_algorithm()
    t_alg = time() - t0
    if val_alg is not None:
        gap = (val_milp - val_alg) / abs(val_milp) * 100
    else:
        gap = None
    results.append({'Алгоритм': 'Шаги 0–3 (наш)', 'Значение': val_alg, 'Время': t_alg,
                    'Разрыв %': gap})

    df = pd.DataFrame(results)
    print("=" * 75)
    print(f"  ЗАДАЧА: N={N}, M={M},  max c@x,  0 <= Ax <= 100,  x ∈ {{0,1}}")
    print("=" * 75)
    print(df.to_string(index=False))
    print("=" * 75)
    print(f"\n  Ускорение относительно прямого MILP: {t_milp / t_alg:.1f}x")
    if gap is not None:
        print(f"  Разрыв с оптимумом: {gap:.2f}%")
    return df


if __name__ == '__main__':
    run_all()

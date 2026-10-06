import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from time import time
import pandas as pd
import scipy.sparse as sp

# HiGHS напрямую (pip install highspy): персистентная модель с warm start базиса.
try:
    import highspy as _hs
    _Highs = _hs.Highs
except ImportError:   # запасной вариант: то же ядро HiGHS, встроенное в scipy
    from scipy.optimize._highspy import _core as _hs
    _Highs = _hs._Highs


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
# Параметры алгоритма (см. раздел 3.2 "Инициализация")
# ============================================================
EPS = 1e-6                    # порог целочисленности ε
RHO_HAT = 0.1                 # ρ̂: ρ_0 = ρ̂ · max|c_i| (привязка к масштабу c)
RHO_0 = RHO_HAT * np.max(np.abs(c_min))   # начальный штрафной коэффициент ρ_0
GAMMA = 0.1                   # γ: нижняя граница веса штрафа ρ·γ
DELTA = 0.1                   # δ: ширина зоны неопределённости вокруг 1/2
BETA = 2.0                    # β: коэффициент увеличения штрафа
RHO_MAX = 1e6 * RHO_0         # потолок штрафа ρ_max (ограничивает и число возмущений)
THETA = 0.1                   # θ: доля переменных, у которых меняется знак d_i
ZETA = 0.05                   # ζ: доля переменных, округляемых на Шаге 2′
K_THRESH = 30                 # K: порог |J| для перехода на Шаг 3
K_MAX = 2 * K_THRESH          # K_max: макс. размер MILP при откате / аварийном входе
B_MAX = 3                     # B_max: макс. число откатов на Шаге 3
STAG_HISTORY = 4              # стагнация: совпадение с x̄ или тремя прошлыми решениями
MILP_TIME_LIMIT = 60.0        # ограничение по времени для MILP, сек
MAX_ITER = 1000               # предохранитель от зацикливания
SEED = 0                      # seed для случайных знаков σ_i и возмущений
VERBOSE = False

# --- Параметры MILP ---
MILP_WARM_START = True        # добавлять отсечение по лучшему решению в MILP
OBJ_CUT_TOL = 1e-6            # допуск для отсечения по целевой

rng = np.random.default_rng(SEED)
STATS = {}


def reset_stats():
    STATS.clear()
    STATS.update({'lp': 0, 'milp': 0, 'perturb': 0, 'dive': 0, 'backtrack': 0,
                  'best_updates': 0, 'lp_iters': 0, 'milp_warm': 0})


def log(*args):
    if VERBOSE:
        print(*args)


class State:
    """Состояние алгоритма: x̄, множества I (зафиксированы) и J (свободны),
    стек H групп фиксаций, текущий штраф ρ и счётчик откатов B."""

    def __init__(self, x_bar, I, J):
        self.x_bar = x_bar
        self.I = I
        self.J = J
        self.H = []
        self.rho = RHO_0
        self.B = 0


# ============================================================
# Сохранение лучшего решения
# ============================================================
class BestSolution:
    """Отслеживает лучшее допустимое бинарное решение, найденное в процессе.

    Даже если основной конвейер не доводит до конца (return false),
    лучшее промежуточное решение сохраняется и возвращается пользователю.
    Проверка выполняется:
      - после LP-релаксации (Шаг 0);
      - после каждой успешной фиксации (Шаг 1 / Шаг 2′);
      - после каждого LP-решения в penalty_step (Шаг 2);
      - после MILP (Шаг 3);
      - после finalize.
    """

    def __init__(self):
        self.x = None
        self.obj = -np.inf       # c @ x (максимизация)
        self.source = None
        self.n_updates = 0

    def try_update(self, x, source):
        """Пробует округлить x до {0,1}, проверить допустимость и обновить рекорд.

        Округление и проверка дёшевы (матрично-векторное умножение A_ub @ x),
        поэтому вызывается часто. Возвращает True при обновлении рекорда.
        """
        x_bin = np.round(x).astype(float)
        # Проверка допустимости: A_ub @ x <= b_ub (нарушение = residual > 1e-6)
        residual = A_ub @ x_bin - b_ub
        if np.any(residual > 1e-6):
            return False

        obj = float(c @ x_bin)
        if obj > self.obj + 1e-12:
            self.x = x_bin.copy()
            self.obj = obj
            self.source = source
            self.n_updates += 1
            STATS['best_updates'] += 1
            log(f"  [best] обновлён: obj={obj:.4f}, источник={source}")
            return True
        return False

    def obj_bound(self):
        """Целевая граница для MILP warm start или None."""
        if self.x is None:
            return None
        return self.obj


# Глобальный экземпляр
best = BestSolution()


# ============================================================
# Персистентная LP-модель (HiGHS): настоящий warm start
# ============================================================
class HighsLP:
    """Одна LP-модель на всех N столбцах, живущая весь прогон алгоритма.

    Фиксация переменной (Шаги 0–1, 2′) — это смена границ столбца
    lo = up = значение; снятие фиксации (откат) — возврат к [0, 1];
    штраф Шага 2 — смена коэффициентов целевой функции. Матрица и
    правые части не пересобираются, а HiGHS сам стартует со старого базиса:
      - после смены границ базис остаётся двойственно допустимым
        (двойственный симплекс);
      - после смены целевой функции базис остаётся прямо допустимым
        (прямой симплекс). Выбор метода HiGHS делает сам.
    Presolve включён только для самого первого (холодного) решения:
    при наличии валидного базиса он только мешает.
    Ограничения lb <= Ax <= ub передаются как есть (M строк, а не 2M).
    """

    def __init__(self):
        self.h = _Highs()
        self.h.setOptionValue('output_flag', False)
        lp = _hs.HighsLp()
        S = sp.csc_matrix(A)
        lp.num_col_, lp.num_row_ = N, M
        lp.col_cost_ = c_min.copy()
        lp.col_lower_ = np.zeros(N)
        lp.col_upper_ = np.ones(N)
        lp.row_lower_ = lb_constr.astype(float).copy()
        lp.row_upper_ = ub_constr.astype(float).copy()
        lp.a_matrix_.format_ = _hs.MatrixFormat.kColwise
        lp.a_matrix_.start_ = S.indptr
        lp.a_matrix_.index_ = S.indices
        lp.a_matrix_.value_ = S.data
        self.h.passModel(lp)
        self.lo = np.zeros(N)       # текущие границы столбцов в модели
        self.up = np.ones(N)
        self.cost = c_min.copy()    # текущие коэффициенты целевой функции
        self.cold = True

    def solve(self, I_arr, J_arr, x_fixed, obj_J):
        """Фиксирует I (по значениям x_fixed), освобождает остальные,
        задаёт целевую c_min (с заменой на obj_J на J) и решает LP.
        Возвращает (успех, вектор x длины N)."""
        lo = np.zeros(N)
        up = np.ones(N)
        if len(I_arr):
            lo[I_arr] = up[I_arr] = x_fixed[I_arr]
        ch = np.flatnonzero((lo != self.lo) | (up != self.up)).astype(np.int32)
        if len(ch):
            self.h.changeColsBounds(len(ch), ch, lo[ch], up[ch])
            self.lo, self.up = lo, up

        cost = c_min.copy()
        cost[J_arr] = obj_J
        ch = np.flatnonzero(cost != self.cost).astype(np.int32)
        if len(ch):
            self.h.changeColsCost(len(ch), ch, cost[ch])
            self.cost = cost

        self.h.setOptionValue('presolve', 'on' if self.cold else 'off')
        self.cold = False
        self.h.run()
        STATS['lp_iters'] += int(self.h.getInfo().simplex_iteration_count)
        if self.h.getModelStatus() != _hs.HighsModelStatus.kOptimal:
            return False, None
        return True, np.array(self.h.getSolution().col_value)


lp_model = None   # создаётся в run_algorithm


# ============================================================
# Вспомогательные функции
# ============================================================
def solve_lp(I, J, x_fixed, mod_c=None):
    """LP: min c_min[J] @ x_J + mod_c @ x_J
    при A_ub[:,J] @ x_J <= b_ub - A_ub[:,I] @ x_I,  0 <= x <= 1.

    Решается в персистентной модели HiGHS (lp_model) со старым базисом.
    Возвращает (x_full, obj) или (None, None) при недопустимости.
    """
    J_arr = np.array(sorted(J), dtype=int)
    I_arr = np.array(sorted(I), dtype=int) if I else np.array([], dtype=int)

    # Проверка ограничений при J = ∅
    if len(J_arr) == 0:
        violation = A_ub @ x_fixed - b_ub
        if np.all(violation <= 1e-8):
            return x_fixed.copy(), float(c_min @ x_fixed)
        return None, None

    obj = c_min[J_arr].copy()
    if mod_c is not None:
        obj = obj + mod_c

    STATS['lp'] += 1
    ok, x = lp_model.solve(I_arr, J_arr, x_fixed, obj)
    if not ok:
        return None, None

    x_full = x_fixed.copy()
    x_full[J_arr] = x[J_arr]
    return x_full, float(obj @ x[J_arr])


def solve_milp(I, J, x_fixed, obj_bound=None):
    """MILP на свободных переменных J. Возвращает (x_full, obj) или (None, None).

    Warm start: если obj_bound задан (лучшее известное значение c @ x),
    добавляется отсечение по целевой функции:
        c @ x >= obj_bound - tol  →  c_min[J] @ x_J <= -obj_bound + tol - c_min[I] @ x_I
    Это отсекает заведомо худшие решения и ускоряет branch-and-bound.
    При недопустимости с отсечением (res.x is None) вызывающий код
    может повторить вызов без отсечения (obj_bound=None).
    """
    J_arr = np.array(sorted(J), dtype=int)
    I_arr = np.array(sorted(I), dtype=int) if I else np.array([], dtype=int)

    if len(J_arr) == 0:
        violation = A_ub @ x_fixed - b_ub
        if np.all(violation <= 1e-8):
            return x_fixed.copy(), float(c_min @ x_fixed)
        return None, None

    b = b_ub.copy()
    if len(I_arr) > 0:
        b = b - A_ub[:, I_arr] @ x_fixed[I_arr]

    STATS['milp'] += 1

    constraints_list = [LinearConstraint(A=A_ub[:, J_arr], ub=b)]

    # Warm start: отсечение по целевой функции
    if obj_bound is not None and MILP_WARM_START and len(J_arr) > 0:
        c_fixed_contrib = float(c_min[I_arr] @ x_fixed[I_arr]) if len(I_arr) > 0 else 0.0
        # c @ x >= obj_bound - tol  →  c_min @ x <= -obj_bound + tol
        # c_min[I] @ x_I + c_min[J] @ x_J <= -obj_bound + tol
        # c_min[J] @ x_J <= -obj_bound + tol - c_min[I] @ x_I
        obj_cut_rhs = -obj_bound + OBJ_CUT_TOL - c_fixed_contrib
        obj_constraint = LinearConstraint(
            A=c_min[J_arr].reshape(1, -1),
            ub=np.array([obj_cut_rhs])
        )
        constraints_list.append(obj_constraint)
        STATS['milp_warm'] += 1

    res = milp(c=c_min[J_arr],
               bounds=Bounds(lb=np.zeros(len(J_arr)), ub=np.ones(len(J_arr))),
               constraints=constraints_list,
               integrality=[1] * len(J_arr),
               options={'time_limit': MILP_TIME_LIMIT})
    if res.x is None:
        return None, None

    x_full = x_fixed.copy()
    x_full[J_arr] = res.x
    # Полный objective с учётом зафиксированных переменных
    obj_full = float(c_min @ x_full)
    return x_full, obj_full


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
    """Пытается зафиксировать кандидатов I_cand, округляя их (LP-1 / LP-2).
    При недопустимости делит I_cand пополам (бисекция множества).
    Возвращает (ok, I, J, x_full, group), где group — реально зафиксированная
    группа индексов (для записи в стек H).
    """
    I_cand = set(I_cand)
    while I_cand:
        x_new = x_fixed.copy()
        for i in I_cand:
            x_new[i] = round(x_vals[i])

        new_I = I | I_cand
        new_J = J - I_cand
        x_test, _ = solve_lp(new_I, new_J, x_new)

        if x_test is not None:
            return True, new_I, new_J, x_test, I_cand

        if len(I_cand) <= 1:
            break

        I_cand, _ = split_I_half(I_cand, x_vals)

    return False, I, J, x_fixed, set()


# ============================================================
# Шаг 0, Шаг 1 и Шаг 2′: процедура округления
# ============================================================
def rounding_step(st, label, eps=EPS, k_thresh=K_THRESH):
    """Шаг 0–1 (метки '1', '2') и Шаг 2′ (метка "2'").

    Метки: '1' — пришли из Шага 0 или Шага 1; '2' — из Шага 2; "2'" — из Шага 2′.
    Кандидаты на округление:
      '1', '2' : почти бинарные компоненты J (|x̄_i − round(x̄_i)| < ε);
      "2'"     : p = max(1, ⌈ζ|J|⌉) компонент J, ближайших к {0,1} (diving).

    Возвращает статус:
      'true'     — J = ∅, найдено бинарное решение;
      'step3'    — |J| <= K, переход на Шаг 3;
      'continue' — фиксация принята, goto Шаг 1 (метка '1');
      'empty'    — кандидатов нет (или фиксация недопустима после бисекции).
    """
    if label == "2'":
        p = max(1, int(np.ceil(ZETA * len(st.J))))
        keyed = sorted((abs(st.x_bar[i] - round(st.x_bar[i])), rng.random(), i)
                       for i in st.J)
        I_cand = set(i for _, _, i in keyed[:p])
        STATS['dive'] += 1
    else:
        I_cand = set(i for i in st.J if is_binary(st.x_bar[i], eps))

    if not I_cand:
        return 'empty'

    ok, new_I, new_J, new_x, group = try_fix(st.I, st.J, st.x_bar, I_cand, st.x_bar)
    if not ok:
        return 'empty'

    st.I, st.J, st.x_bar = new_I, new_J, new_x
    st.H.append(group)

    # Сохранение лучшего решения: после фиксации проверяем,
    # даёт ли округление текущего x_bar допустимое бинарное решение
    best.try_update(st.x_bar, f'round[{label}]')

    if not st.J:
        return 'true'
    if len(st.J) <= k_thresh:
        return 'step3'
    return 'continue'


# ============================================================
# Шаг 2: линеаризованный штраф с зоной неопределённости и возмущением
# ============================================================
def penalty_step(st, eps=EPS):
    """Шаг 2. Решает (LP-3) с линеаризованным штрафом.

    Меняется только целевая функция при тех же фиксациях, поэтому
    каждый LP-3 стартует с базиса предыдущего решения.

    Каждая итерация без кандидатов на округление умножает ρ на β:
      - решение сдвинулось: x̄ ← x;
      - стагнация: x̄ не меняется, у ⌈θ|J|⌉ самых неопределённых переменных
        в следующем LP-3 меняется знак d_i (возмущение). Само по себе
        возмущение при малом ρ бесполезно (штраф слабее цели c), поэтому
        оно сопровождается ростом ρ.
    Отдельного счётчика возмущений нет: их число ограничено потолком ρ_max
    (за одно посещение не больше ⌊log_β(ρ_max/ρ_0)⌋ + 1 итераций).

    Возвращает:
      'candidates' — появились почти бинарные переменные (st.x_bar обновлён,
                     ρ ← max(ρ_0, ρ/β)), далее Шаг 1 с меткой '2';
      'exhausted'  — ρ превысил ρ_max (в том числе при стагнации с
                     возмущениями), далее Шаг 2′ (ρ сбрасывается в ρ_0).
    """
    J_arr = np.array(sorted(st.J), dtype=int)
    x_bar = st.x_bar.copy()
    rho = st.rho
    history = [x_bar[J_arr].copy()]          # x̄_J и прошлые решения Шага 2
    flip = None                              # индексы с изменённым знаком d_i

    for _ in range(MAX_ITER):
        xb = x_bar[J_arr]
        slope = 1 - 2 * xb

        # ρ_i = ρ · ((1 − 2·x̄_i)² + γ)
        rho_i = rho * (slope ** 2 + GAMMA)

        # σ_i = sign(1 − 2·x̄_i) при |1 − 2·x̄_i| >= δ, иначе случайный знак
        sigma = np.sign(slope)
        undecided = np.abs(slope) < DELTA
        sigma[undecided] = rng.choice([-1.0, 1.0], size=int(undecided.sum()))

        # d_i = σ_i · max{|1 − 2·x̄_i|, δ}
        d = sigma * np.maximum(np.abs(slope), DELTA)
        if flip is not None:                 # возмущение действует на одно решение
            d[flip] *= -1.0

        # (LP-3): c_J @ x_J + Σ ρ_i d_i (x_i − x̄_i)  (константа опущена)
        x_new, _ = solve_lp(st.I, st.J, x_bar, mod_c=rho_i * d)
        if x_new is None:
            break
        xn = x_new[J_arr]

        # Сохранение лучшего решения: промежуточные LP-решения
        # могут содержать почти-бинарные компоненты
        best.try_update(x_new, 'penalty')

        # Кандидаты на округление I' ≠ ∅ → возвращаемся на Шаг 1
        if np.any(np.abs(xn - np.round(xn)) < eps):
            st.x_bar = x_new
            st.rho = max(RHO_0, rho / BETA)
            return 'candidates'

        # Стагнация: x совпадает с x̄ или с одним из трёх прошлых решений
        stagnation = any(np.max(np.abs(xn - h)) < eps
                         for h in history[-STAG_HISTORY:])

        if not stagnation:
            # Решение сдвинулось: x̄ ← x
            x_bar = x_new
            history.append(xn.copy())
            flip = None
        else:
            # Стагнация: x̄ не меняется, возмущаем знаки d_i у ⌈θ|J|⌉ самых
            # неопределённых переменных (наименьшие |1 − 2·x̄_i|, при
            # равенстве — случайные); флип применится в следующем LP-3
            STATS['perturb'] += 1
            m = max(1, int(np.ceil(THETA * len(J_arr))))
            order = np.lexsort((rng.random(len(xb)), np.abs(slope)))
            flip = order[:m]

        # В обоих случаях штраф растёт: ρ ← βρ; потолок ρ_max ограничивает
        # и число попыток возмущения
        rho *= BETA
        if rho > RHO_MAX:
            break

    st.x_bar = x_bar
    st.rho = RHO_0
    return 'exhausted'


# ============================================================
# Шаг 3: точное решение MILP с откатом
# ============================================================
def exact_step(st):
    """Шаг 3. Решает MILP на J с warm start (отсечение по лучшему решению).

    При недопустимости MILP с отсечением — retry без отсечения:
      - если retry успешен — решение найдено;
      - если retry недопустим — откат последней группы из стека H
        (не более B_max раз, при условии |J| <= K_max).

    Возвращает (x_full, obj) или (None, None).
    """
    obj_bound = best.obj_bound()

    while True:
        x_full, obj = solve_milp(st.I, st.J, st.x_bar, obj_bound=obj_bound)
        if x_full is not None:
            best.try_update(x_full, 'milp')
            return x_full, obj

        # MILP недопустим. Если использовалось отсечение — retry без него
        if obj_bound is not None:
            log("  MILP недопустим с отсечением, retry без warm start")
            obj_bound = None
            continue

        # Без отсечения тоже недопустим — откат
        if st.B >= B_MAX or not st.H:
            return None, None
        group = st.H[-1]
        if len(st.J) + len(group) > K_MAX:
            return None, None

        st.H.pop()
        st.B += 1
        STATS['backtrack'] += 1
        st.I = st.I - group
        st.J = st.J | group
        log(f"  откат: |J| = {len(st.J)}, B = {st.B}")
        # После отката обновляем отсечение (множество J изменилось)
        obj_bound = best.obj_bound()


def finalize(x):
    """Округляет решение до {0,1} и проверяет допустимость.

    Проверяет нарушение ограничений: A_ub @ x - b_ub > 1e-6.
    (Отрицательный residual означает, что ограничение выполнено с запасом,
    а не нарушено — поэтому проверка только сверху.)
    """
    x = np.round(x)
    if np.any(A_ub @ x - b_ub > 1e-6):
        return None
    return x


# ============================================================
# Главный цикл алгоритма
# ============================================================
def run_algorithm(k_thresh=K_THRESH, eps=EPS, seed=SEED):
    """Основной алгоритм: Шаг 0 → Шаг 1 ↔ Шаг 2 → Шаг 2′ → Шаг 3.

    Возвращает (x, c @ x, info). При неудаче основного конвейера
    возвращает лучшее найденное промежуточное решение (если есть),
    иначе x = None.
    info: z_lp — LP-граница, gap — оценка отклонения от оптимума,
          stats, best_source — источник лучшего решения.
    """
    global rng, best, lp_model, RHO_0, RHO_MAX
    rng = np.random.default_rng(seed)
    reset_stats()
    best = BestSolution()
    RHO_0 = RHO_HAT * np.max(np.abs(c_min))   # масштаб штрафа от текущей c
    RHO_MAX = 1e6 * RHO_0
    lp_model = HighsLP()
    info = {'z_lp': None, 'gap': None, 'stats': STATS, 'best_source': None}

    # Шаг 0: LP-релаксация, нижняя граница z_LP
    I, J = set(), set(range(N))
    x0, z_lp = solve_lp(I, J, np.zeros(N))
    if x0 is None:
        return None, None, info
    info['z_lp'] = z_lp

    # Проверка лучшего решения уже на LP-релаксации
    best.try_update(x0, 'lp_relaxation')

    st = State(x0, I, J)
    label, action = '1', 'round'
    if len(st.J) <= k_thresh:
        action = 'step3'
    result = None

    for _ in range(MAX_ITER):
        if action == 'round':
            status = rounding_step(st, label, eps=eps, k_thresh=k_thresh)
            log(f"round[{label}] -> {status}, |J| = {len(st.J)}")
            if status == 'true':
                result = st.x_bar
                break
            if status == 'step3':
                action = 'step3'
            elif status == 'continue':
                label = '1'
            elif label == '1':                       # кандидатов нет
                action = 'penalty'
            elif label == '2':                       # фиксация недопустима
                label = "2'"
            else:                                    # метка 2′, ничего не округлено
                if len(st.J) <= K_MAX:
                    action = 'step3'
                else:
                    break                            # return false
        elif action == 'penalty':
            res = penalty_step(st, eps=eps)
            log(f"penalty -> {res}, rho = {st.rho:.3g}, |J| = {len(st.J)}")
            label = '2' if res == 'candidates' else "2'"
            action = 'round'
        elif action == 'step3':
            result, _ = exact_step(st)
            break

    # Если основной конвейер нашёл решение — финализируем
    if result is not None:
        x = finalize(result)
        if x is not None:
            best.try_update(x, 'finalize')

    # Возвращаем лучшее найденное решение
    # (даже если конвейер не дошёл до конца)
    if best.x is not None:
        info['gap'] = (c_min @ best.x - z_lp) / max(1.0, abs(z_lp))
        info['best_source'] = best.source
        return best.x, best.obj, info

    return None, None, info


# ============================================================
# Подмена данных задачи и пример стагнации из документа
# ============================================================
def set_problem(c_new, A_new, lb_new, ub_new):
    """Подменяет данные задачи: max c @ x, lb <= A x <= ub, x ∈ {0,1}^N."""
    global N, M, c, A, lb_constr, ub_constr, c_min, A_ub, b_ub
    c = np.asarray(c_new, dtype=float)
    A = np.asarray(A_new, dtype=float)
    lb_constr = np.asarray(lb_new, dtype=float)
    ub_constr = np.asarray(ub_new, dtype=float)
    M, N = A.shape
    c_min = -c
    A_ub = np.vstack([A, -A])
    b_ub = np.concatenate([ub_constr, -lb_constr])


def demo_stagnation(k_thresh=0):
    """Пример из замечаний 3–4 документа:
         min x1 + 2·x2,  x1 + x2 >= 1,3,  x ∈ {0,1}^2   (оптимум (1,1), значение 3).
    LP-оптимум (1; 0,3) — стагнация: линеаризация тянет x2 к недопустимому 0.
    Смена знака d_2 уводит решение в (1,1) лишь когда штраф перевесит цель:
    при весах алгоритма и γ = 0,1 это ρ > ≈19; из ρ_0 = 0,2 с β = 2 — 8-я итерация.
    Поэтому возмущение должно сопровождаться ростом ρ.
    """
    global K_MAX
    saved = (c, A, lb_constr, ub_constr, K_MAX)
    set_problem(c_new=[-1.0, -2.0], A_new=[[1.0, 1.0]], lb_new=[1.3], ub_new=[2.0])
    K_MAX = k_thresh
    try:
        x, val, info = run_algorithm(k_thresh=k_thresh)
    finally:
        set_problem(*saved[:4])
        K_MAX = saved[4]
    print("Пример стагнации (замечания 3–4):")
    print(f"  решение: {None if x is None else x.tolist()}, c@x = {val}")
    print(f"  статистика: {info['stats']}")
    return x, val, info


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

    lp_gap = (
        (val_lp - val_milp) / abs(val_milp) * 100
        if abs(val_milp) > 1e-12 else None
    )
    results.append({'Алгоритм': 'LP-релаксация', 'Значение': val_lp, 'Время': t_lp,
                    'Разрыв %': lp_gap, 'gap к LP %': None})
    results.append({'Алгоритм': 'Прямое MILP', 'Значение': val_milp, 'Время': t_milp,
                    'Разрыв %': 0.0, 'gap к LP %': None})

    # Наш алгоритм
    t0 = time()
    x_alg, val_alg, info = run_algorithm()
    t_alg = time() - t0
    if val_alg is not None:
        gap = (
            (val_milp - val_alg) / abs(val_milp) * 100
            if abs(val_milp) > 1e-12 else None
        )
        gap_lp = info['gap'] * 100
    else:
        gap = None
        gap_lp = None
    results.append({'Алгоритм': 'Шаги 0–3 (наш)', 'Значение': val_alg, 'Время': t_alg,
                    'Разрыв %': gap, 'gap к LP %': gap_lp})

    df = pd.DataFrame(results)
    print("=" * 75)
    print(f"  ЗАДАЧА: N={N}, M={M},  max c@x,  0 <= Ax <= 100,  x ∈ {{0,1}}")
    print("=" * 75)
    print(df.to_string(index=False))
    print("=" * 75)
    if x_alg is None:
        print("\n  Алгоритм не нашёл допустимого решения (return false)")
    else:
        print(f"\n  Ускорение относительно прямого MILP: {t_milp / max(t_alg, 1e-9):.1f}x")
        if gap is not None:
            print(f"  Разрыв с оптимумом: {gap:.2f}%")
        print(f"  gap к LP-границе: {gap_lp:.2f}%")
        print(f"  Источник решения: {info.get('best_source', '?')}")
    print(f"  Статистика: {info['stats']}")
    print(f"  LP: {info['stats']['lp']} решений, {info['stats']['lp_iters']} итераций симплекса")
    return df


if __name__ == '__main__':
    run_all()
    print()
    demo_stagnation()

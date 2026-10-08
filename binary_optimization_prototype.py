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

# Ограничения: lb <= Ax <= ub  ->  A_ub @ x <= b_ub
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

# --- Параметр логирования ---
# LOG_ENABLED = True  — детальный лог каждого шага и итерации
# LOG_ENABLED = False — логирование полностью выключено, нулевые накладные расходы
LOG_ENABLED = False

# --- Параметры MILP ---
MILP_WARM_START = True        # добавлять отсечение по лучшему решению в MILP
OBJ_CUT_TOL = 1e-6            # допуск для отсечения по целевой

rng = np.random.default_rng(SEED)
STATS = {}

# Счётчики LP по шагам
_lp_step1 = 0
_lp_step2 = 0
_lp_step2p = 0
_lp_iters_step1 = 0
_lp_iters_step2 = 0
_lp_iters_step2p = 0
_current_step_label = 'step1'


def reset_stats():
    global _lp_step1, _lp_step2, _lp_step2p, _lp_iters_step1, _lp_iters_step2, _lp_iters_step2p
    STATS.clear()
    STATS.update({'lp': 0, 'milp': 0, 'perturb': 0, 'dive': 0, 'backtrack': 0,
                  'best_updates': 0, 'lp_iters': 0, 'milp_warm': 0})
    _lp_step1 = _lp_step2 = _lp_step2p = 0
    _lp_iters_step1 = _lp_iters_step2 = _lp_iters_step2p = 0


# ============================================================
# Логирование: три уровня, все за guarded LOG_ENABLED
# ============================================================
_log_indent = 0


def log_step(name, **kwargs):
    """Вход в шаг. Печатает название, |I|, |J|, ρ и другие параметры."""
    global _log_indent
    if not LOG_ENABLED:
        return
    prefix = '  ' * _log_indent
    parts = [f"{name}"]
    for k, v in kwargs.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        elif isinstance(v, set):
            parts.append(f"|{k}|={len(v)}")
        else:
            parts.append(f"{k}={v}")
    print(f"{prefix}{' '.join(parts)}")
    _log_indent += 1


def log_iter(iter_num, **kwargs):
    """Итерация внутри шага."""
    if not LOG_ENABLED:
        return
    prefix = '  ' * _log_indent
    parts = [f"[{iter_num}]"]
    for k, v in kwargs.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        elif isinstance(v, (list, np.ndarray)):
            parts.append(f"{k}={np.array2string(np.asarray(v), threshold=6, precision=4)}")
        elif isinstance(v, set):
            parts.append(f"|{k}|={len(v)}")
        else:
            parts.append(f"{k}={v}")
    print(f"{prefix}{' '.join(parts)}")


def log_detail(msg, **kwargs):
    """Детали внутри итерации."""
    if not LOG_ENABLED:
        return
    prefix = '  ' * (_log_indent + 1)
    if kwargs:
        parts = [msg]
        for k, v in kwargs.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4g}")
            elif isinstance(v, (list, np.ndarray)):
                parts.append(f"{k}={np.array2string(np.asarray(v), threshold=6, precision=4)}")
            elif isinstance(v, set):
                parts.append(f"|{k}|={len(v)}")
            else:
                parts.append(f"{k}={v}")
        print(f"{prefix}{' '.join(parts)}")
    else:
        print(f"{prefix}{msg}")


def log_step_end(name, **kwargs):
    """Выход из шага."""
    global _log_indent
    if not LOG_ENABLED:
        return
    _log_indent = max(0, _log_indent - 1)
    prefix = '  ' * _log_indent
    parts = [f"-> {name}"]
    for k, v in kwargs.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        elif isinstance(v, set):
            parts.append(f"|{k}|={len(v)}")
        else:
            parts.append(f"{k}={v}")
    print(f"{prefix}{' '.join(parts)}")


def log(*args):
    """Простая лог-функция (для совместимости со старым кодом)."""
    if not LOG_ENABLED:
        return
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
    """

    def __init__(self):
        self.x = None
        self.obj = -np.inf       # c @ x (максимизация)
        self.source = None
        self.n_updates = 0

    def try_update(self, x, source):
        """Пробует округлить x до {0,1}, проверить допустимость и обновить рекорд."""
        x_bin = np.round(x).astype(float)
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
            log_detail(f"[best] обновлён: obj={obj:.4f}, источник={source}")
            return True
        return False

    def obj_bound(self):
        """Целевая граница для MILP warm start или None."""
        if self.x is None:
            return None
        return self.obj


best = BestSolution()


# ============================================================
# Персистентная LP-модель (HiGHS)
# ============================================================
class HighsLP:
    """Одна LP-модель на всех N столбцах, живущая весь прогон."""

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
        self.lo = np.zeros(N)
        self.up = np.ones(N)
        self.cost = c_min.copy()
        self.cold = True

    def solve(self, I_arr, J_arr, x_fixed, obj_J):
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
        iters = int(self.h.getInfo().simplex_iteration_count)
        STATS['lp_iters'] += iters
        _track_lp_iters(iters)
        if self.h.getModelStatus() != _hs.HighsModelStatus.kOptimal:
            return False, None
        return True, np.array(self.h.getSolution().col_value)


lp_model = None


def _track_lp_iters(iters):
    """Учитывает итерации симплекса по шагам."""
    global _lp_iters_step1, _lp_iters_step2, _lp_iters_step2p
    if _current_step_label == 'step1':
        _lp_iters_step1 += iters
    elif _current_step_label == 'step2':
        _lp_iters_step2 += iters
    elif _current_step_label == "step2'":
        _lp_iters_step2p += iters


def _track_lp():
    """Учитывает число LP-решений по шагам."""
    global _lp_step1, _lp_step2, _lp_step2p
    if _current_step_label == 'step1':
        _lp_step1 += 1
    elif _current_step_label == 'step2':
        _lp_step2 += 1
    elif _current_step_label == "step2'":
        _lp_step2p += 1


# ============================================================
# Вспомогательные функции
# ============================================================
def solve_lp(I, J, x_fixed, mod_c=None):
    """LP: min c_min[J] @ x_J + mod_c @ x_J
    при A_ub[:,J] @ x_J <= b_ub - A_ub[:,I] @ x_I,  0 <= x <= 1.
    """
    global _current_step_label
    J_arr = np.array(sorted(J), dtype=int)
    I_arr = np.array(sorted(I), dtype=int) if I else np.array([], dtype=int)

    if len(J_arr) == 0:
        violation = A_ub @ x_fixed - b_ub
        if np.all(violation <= 1e-8):
            return x_fixed.copy(), float(c_min @ x_fixed)
        return None, None

    obj = c_min[J_arr].copy()
    if mod_c is not None:
        obj = obj + mod_c

    STATS['lp'] += 1
    _track_lp()
    ok, x = lp_model.solve(I_arr, J_arr, x_fixed, obj)
    if not ok:
        return None, None

    x_full = x_fixed.copy()
    x_full[J_arr] = x[J_arr]
    return x_full, float(obj @ x[J_arr])


def solve_milp(I, J, x_fixed, obj_bound=None):
    """MILP на свободных переменных J."""
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

    if obj_bound is not None and MILP_WARM_START and len(J_arr) > 0:
        c_fixed_contrib = float(c_min[I_arr] @ x_fixed[I_arr]) if len(I_arr) > 0 else 0.0
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
    obj_full = float(c_min @ x_full)
    return x_full, obj_full


def is_binary(x, eps=EPS):
    return abs(x - round(x)) < eps


def split_I_half(I_cand, x_vals):
    deviations = [(i, abs(x_vals[i] - round(x_vals[i]))) for i in I_cand]
    deviations.sort(key=lambda t: t[1])
    half = max(1, len(deviations) // 2)
    keep = set(i for i, _ in deviations[:half])
    moved = set(i for i, _ in deviations[half:])
    return keep, moved


def try_fix(I, J, x_fixed, I_cand, x_vals):
    """Пытается зафиксировать кандидатов I_cand. При недопустимости — бисекция."""
    I_cand = set(I_cand)
    bisec_count = 0
    while I_cand:
        x_new = x_fixed.copy()
        for i in I_cand:
            x_new[i] = round(x_vals[i])

        new_I = I | I_cand
        new_J = J - I_cand
        x_test, _ = solve_lp(new_I, new_J, x_new)

        if x_test is not None:
            if LOG_ENABLED and bisec_count > 0:
                log_detail(f"бисекция: {bisec_count} шагов деления, осталось {len(I_cand)} из {len(I_cand) << bisec_count}")
            return True, new_I, new_J, x_test, I_cand

        if len(I_cand) <= 1:
            break

        I_cand, _ = split_I_half(I_cand, x_vals)
        bisec_count += 1

    return False, I, J, x_fixed, set()


# ============================================================
# Шаг 1 и Шаг 2′: процедура округления
# ============================================================
def rounding_step(st, label, eps=EPS, k_thresh=K_THRESH):
    """Шаг 1 (метки '1', '2') и Шаг 2′ (метка "2'")."""
    global _current_step_label
    _current_step_label = "step2'" if label == "2'" else "step1"

    log_step(f"Шаг 1[{label}]", I=st.I, J=st.J, rho=st.rho, label=label)

    if label == "2'":
        p = max(1, int(np.ceil(ZETA * len(st.J))))
        keyed = sorted((abs(st.x_bar[i] - round(st.x_bar[i])), rng.random(), i)
                       for i in st.J)
        I_cand = set(i for _, _, i in keyed[:p])
        STATS['dive'] += 1
        if LOG_ENABLED:
            log_detail(f"diving: выбрано {p} переменных из {len(st.J)}")
            log_detail(f"индексы: {sorted(I_cand)}")
            for i in sorted(I_cand):
                log_detail(f"  x[{i}]={st.x_bar[i]:.4f} -> {round(st.x_bar[i])} "
                           f"(откл={abs(st.x_bar[i]-round(st.x_bar[i])):.4f})")
    else:
        I_cand = set(i for i in st.J if is_binary(st.x_bar[i], eps))

    if not I_cand:
        log_detail("кандидатов на округление не найдено")
        log_step_end(f"empty (метка {label})")
        return 'empty'

    if LOG_ENABLED:
        log_detail(f"кандидатов: {len(I_cand)}")
        for i in sorted(I_cand):
            log_detail(f"  x[{i}]={st.x_bar[i]:.6f} -> {round(st.x_bar[i])}")

    ok, new_I, new_J, new_x, group = try_fix(st.I, st.J, st.x_bar, I_cand, st.x_bar)
    if not ok:
        log_detail("фиксация недопустима даже после бисекции")
        log_step_end(f"empty (метка {label}, фиксация неудачна)")
        return 'empty'

    st.I, st.J, st.x_bar = new_I, new_J, new_x
    st.H.append(group)

    updated = best.try_update(st.x_bar, f'round[{label}]')
    if LOG_ENABLED and updated:
        log_detail(f"рекорд обновлён после фиксации: obj={best.obj:.4f}")

    if not st.J:
        log_detail("J = пусто, найдено бинарное решение")
        log_step_end("true")
        return 'true'
    if len(st.J) <= k_thresh:
        log_detail(f"|J|={len(st.J)} <= K={k_thresh}, переход на Шаг 3")
        log_step_end("step3")
        return 'step3'
    log_detail(f"фиксация принята: |I|={len(st.I)}, |J|={len(st.J)}")
    log_step_end("continue")
    return 'continue'


# ============================================================
# Шаг 2: линеаризованный штраф с зоной неопределённости и возмущением
# ============================================================
def penalty_step(st, eps=EPS):
    """Шаг 2. Решает (LP-3) с линеаризованным штрафом."""
    global _current_step_label
    _current_step_label = "step2"

    log_step("Шаг 2 (штраф)", I=st.I, J=st.J, rho=st.rho, rho_max=RHO_MAX)

    J_arr = np.array(sorted(st.J), dtype=int)
    x_bar = st.x_bar.copy()
    rho = st.rho
    history = [x_bar[J_arr].copy()]
    flip = None

    for it in range(MAX_ITER):
        xb = x_bar[J_arr]
        slope = 1 - 2 * xb

        rho_i = rho * (slope ** 2 + GAMMA)

        sigma = np.sign(slope)
        undecided = np.abs(slope) < DELTA
        sigma[undecided] = rng.choice([-1.0, 1.0], size=int(undecided.sum()))

        d = sigma * np.maximum(np.abs(slope), DELTA)
        if flip is not None:
            d[flip] *= -1.0

        n_frac = int(np.sum(np.abs(xb - np.round(xb)) >= eps))
        n_near = len(J_arr) - n_frac
        n_undec = int(undecided.sum())

        if LOG_ENABLED:
            log_iter(it, rho=rho, n_frac=n_frac, n_near=n_near, n_undec=n_undec)
            log_detail(f"наклоны d_i: min={d.min():.4g}, max={d.max():.4g}, "
                        f"средн={d.mean():.4g}")
            if flip is not None:
                log_detail(f"возмущение активно: {len(flip)} переменных, "
                           f"индексы={J_arr[flip][:10].tolist()}"
                           f"{'...' if len(flip) > 10 else ''}")

        x_new, _ = solve_lp(st.I, st.J, x_bar, mod_c=rho_i * d)
        if x_new is None:
            log_detail("LP недопустима")
            break
        xn = x_new[J_arr]

        best.try_update(x_new, 'penalty')

        shift = float(np.max(np.abs(xn - xb))) if len(xn) > 0 else 0.0
        n_frac_new = int(np.sum(np.abs(xn - np.round(xn)) >= eps))

        if LOG_ENABLED:
            log_detail(f"LP решён: сдвиг={shift:.6f}, дробных стало={n_frac_new}")

        # Кандидаты на округление
        cand_mask = np.abs(xn - np.round(xn)) < eps
        if np.any(cand_mask):
            cand_idx = J_arr[cand_mask]
            if LOG_ENABLED:
                log_detail(f"найдено кандидатов: {len(cand_idx)}")
                for ci in cand_idx[:5]:
                    log_detail(f"  x[{ci}]={x_new[ci]:.6f} -> {round(x_new[ci])}")
                if len(cand_idx) > 5:
                    log_detail(f"  ... и ещё {len(cand_idx)-5}")
            st.x_bar = x_new
            st.rho = max(RHO_0, rho / BETA)
            log_detail(f"возврат на Шаг 1, rho сброшен до {st.rho:.4g}")
            log_step_end("candidates")
            return 'candidates'

        # Стагнация
        stagnation = any(np.max(np.abs(xn - h)) < eps
                         for h in history[-STAG_HISTORY:])

        if not stagnation:
            x_bar = x_new
            history.append(xn.copy())
            flip = None
            if LOG_ENABLED:
                log_detail("решение сдвинулось, x̄ обновлён")
        else:
            STATS['perturb'] += 1
            m = max(1, int(np.ceil(THETA * len(J_arr))))
            order = np.lexsort((rng.random(len(xb)), np.abs(slope)))
            flip = order[:m]
            if LOG_ENABLED:
                log_detail(f"СТАГНАЦИЯ: возмущение {m} переменных")
                for fi in flip[:5]:
                    log_detail(f"  x[{J_arr[fi]}]={x_bar[J_arr[fi]]:.4f} "
                               f"|d|={abs(slope[fi]):.4f} -> знак сменён")
                if len(flip) > 5:
                    log_detail(f"  ... и ещё {len(flip)-5}")

        rho *= BETA
        if LOG_ENABLED:
            log_detail(f"rho <- {rho:.4g}")
        if rho > RHO_MAX:
            log_detail(f"rho > rho_max, штраф исчерпан")
            break

    st.x_bar = x_bar
    st.rho = RHO_0
    log_step_end("exhausted")
    return 'exhausted'


# ============================================================
# Шаг 3: точное решение MILP с откатом
# ============================================================
def exact_step(st):
    """Шаг 3. Решает MILP на J с warm start."""
    log_step("Шаг 3 (MILP)", I=st.I, J=st.J, B=st.B, B_max=B_MAX,
             stack=len(st.H))

    obj_bound = best.obj_bound()
    attempt = 0

    while True:
        attempt += 1
        if LOG_ENABLED:
            log_iter(attempt, J=st.J, with_cut=(obj_bound is not None))
            log_detail(f"размер MILP: |J|={len(st.J)}")
            if obj_bound is not None:
                log_detail(f"отсечение: c@x >= {obj_bound:.4f}")
            else:
                log_detail("без отсечения")

        x_full, obj = solve_milp(st.I, st.J, st.x_bar, obj_bound=obj_bound)
        if x_full is not None:
            best.try_update(x_full, 'milp')
            if LOG_ENABLED:
                log_detail(f"MILP решён: c@x={float(c@x_full):.4f}")
            log_step_end("решение найдено")
            return x_full, obj

        if obj_bound is not None:
            log_detail("MILP недопустим с отсечением, retry без него")
            obj_bound = None
            continue

        if st.B >= B_MAX or not st.H:
            log_detail(f"откатов: {st.B}/{B_MAX}, стек пуст — сдаёмся")
            log_step_end("нет решения")
            return None, None

        group = st.H[-1]
        if len(st.J) + len(group) > K_MAX:
            log_detail(f"|J|+|group|={len(st.J)+len(group)} > K_max={K_MAX}")
            log_step_end("нет решения (K_max)")
            return None, None

        st.H.pop()
        st.B += 1
        STATS['backtrack'] += 1
        st.I = st.I - group
        st.J = st.J | group
        log_detail(f"откат {st.B}/{B_MAX}: снято {len(group)} фиксаций, "
                   f"|J|={len(st.J)}")
        obj_bound = best.obj_bound()


def finalize(x):
    x = np.round(x)
    if np.any(A_ub @ x - b_ub > 1e-6):
        return None
    return x


# ============================================================
# Главный цикл алгоритма
# ============================================================
def run_algorithm(k_thresh=K_THRESH, eps=EPS, seed=SEED):
    """Основной алгоритм: Шаг 1 <-> Шаг 2 -> Шаг 2′ -> Шаг 3."""
    global rng, best, lp_model, RHO_0, RHO_MAX, _current_step_label
    rng = np.random.default_rng(seed)
    reset_stats()
    best = BestSolution()
    RHO_0 = RHO_HAT * np.max(np.abs(c_min))
    RHO_MAX = 1e6 * RHO_0
    lp_model = HighsLP()
    _current_step_label = 'step1'
    info = {'z_lp': None, 'gap': None, 'stats': STATS, 'best_source': None}

    # Шаг 1, первая итерация: LP-релаксация (I = пусто, J = {0..n-1})
    log_step("Шаг 1[init] — LP-релаксация")
    I, J = set(), set(range(N))
    x0, z_lp = solve_lp(I, J, np.zeros(N))
    if x0 is None:
        log_step_end("LP недопустима")
        return None, None, info
    info['z_lp'] = z_lp
    log_detail(f"z_LP = {-z_lp:.4f}")

    n_frac = int(np.sum(np.abs(x0 - np.round(x0)) >= eps))
    log_detail(f"LP-релаксация: дробных={n_frac}, почти целых={N - n_frac}")

    best.try_update(x0, 'lp_relaxation')
    log_step_end("LP-релаксация готова")

    st = State(x0, I, J)
    label, action = '1', 'round'
    if len(st.J) <= k_thresh:
        action = 'step3'
    result = None

    for main_iter in range(MAX_ITER):
        if LOG_ENABLED:
            log_iter(main_iter, action=action, label=label, I=st.I, J=st.J, rho=st.rho)

        if action == 'round':
            status = rounding_step(st, label, eps=eps, k_thresh=k_thresh)
            if status == 'true':
                result = st.x_bar
                break
            if status == 'step3':
                action = 'step3'
            elif status == 'continue':
                label = '1'
            elif label == '1':
                action = 'penalty'
            elif label == '2':
                label = "2'"
            else:
                if len(st.J) <= K_MAX:
                    action = 'step3'
                else:
                    break
        elif action == 'penalty':
            res = penalty_step(st, eps=eps)
            label = '2' if res == 'candidates' else "2'"
            action = 'round'
        elif action == 'step3':
            result, _ = exact_step(st)
            break

    # Финализация
    if result is not None:
        x_bin = finalize(result)
        if x_bin is not None:
            best.try_update(x_bin, 'finalize')

    if best.x is not None:
        info['gap'] = (float(c_min @ best.x) - info['z_lp']) / max(1.0, abs(info['z_lp']))
        info['best_source'] = best.source
        # Сводка LP по шагам
        STATS['lp_step1'] = _lp_step1
        STATS['lp_step2'] = _lp_step2
        STATS['lp_step2p'] = _lp_step2p
        STATS['lp_iters_step1'] = _lp_iters_step1
        STATS['lp_iters_step2'] = _lp_iters_step2
        STATS['lp_iters_step2p'] = _lp_iters_step2p
        return best.x, best.obj, info

    STATS['lp_step1'] = _lp_step1
    STATS['lp_step2'] = _lp_step2
    STATS['lp_step2p'] = _lp_step2p
    STATS['lp_iters_step1'] = _lp_iters_step1
    STATS['lp_iters_step2'] = _lp_iters_step2
    STATS['lp_iters_step2p'] = _lp_iters_step2p
    return None, None, info


# ============================================================
# Подмена данных задачи и пример стагнации
# ============================================================
def set_problem(c_new, A_new, lb_new, ub_new):
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
    global K_MAX
    saved = (c, A, lb_constr, ub_constr, K_MAX)
    set_problem(c_new=[-1.0, -2.0], A_new=[[1.0, 1.0]], lb_new=[1.3], ub_new=[2.0])
    K_MAX = k_thresh
    try:
        x, val, info = run_algorithm(k_thresh=k_thresh)
    finally:
        set_problem(*saved[:4])
        K_MAX = saved[4]
    print("Пример стагнации (замечания 3-4):")
    print(f"  решение: {None if x is None else x.tolist()}, c@x = {val}")
    print(f"  статистика: {info['stats']}")
    return x, val, info


# ============================================================
# Сравнение с эталонами
# ============================================================
def run_all():
    results = []

    t0 = time()
    res_lp = linprog(c=c_min, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), method='highs')
    t_lp = time() - t0
    val_lp = c @ res_lp.x

    t0 = time()
    res_milp = milp(c=c_min, bounds=Bounds(0, 1),
                    constraints=LinearConstraint(A=A_ub, ub=b_ub),
                    integrality=[1] * N)
    t_milp = time() - t0
    val_milp = c @ res_milp.x

    lp_gap = ((val_lp - val_milp) / abs(val_milp) * 100
              if abs(val_milp) > 1e-12 else None)
    results.append({'Алгоритм': 'LP-релаксация', 'Значение': val_lp, 'Время': t_lp,
                    'Разрыв %': lp_gap, 'gap к LP %': None})
    results.append({'Алгоритм': 'Прямое MILP', 'Значение': val_milp, 'Время': t_milp,
                    'Разрыв %': 0.0, 'gap к LP %': None})

    t0 = time()
    x_alg, val_alg, info = run_algorithm()
    t_alg = time() - t0
    if val_alg is not None:
        gap = ((val_milp - val_alg) / abs(val_milp) * 100
               if abs(val_milp) > 1e-12 else None)
        gap_lp = info['gap'] * 100
    else:
        gap = None
        gap_lp = None
    results.append({'Алгоритм': 'Шаги 1-3 (наш)', 'Значение': val_alg, 'Время': t_alg,
                    'Разрыв %': gap, 'gap к LP %': gap_lp})

    df = pd.DataFrame(results)
    print("=" * 75)
    print(f"  ЗАДАЧА: N={N}, M={M},  max c@x,  0 <= Ax <= 100,  x in {{0,1}}")
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
    s = info['stats']
    print(f"  Статистика: {s}")
    print(f"  LP: {s['lp']} решений, {s['lp_iters']} итераций симплекса")
    print(f"  LP по шагам: ш1={s.get('lp_step1',0)}, ш2={s.get('lp_step2',0)}, "
          f"ш2'={s.get('lp_step2p',0)}")
    print(f"  Итер. симпл. по шагам: ш1={s.get('lp_iters_step1',0)}, "
          f"ш2={s.get('lp_iters_step2',0)}, ш2'={s.get('lp_iters_step2p',0)}")
    return df


if __name__ == '__main__':
    print("=== LOG_ENABLED = False (быстрый прогон) ===")
    LOG_ENABLED = False
    run_all()
    print()
    demo_stagnation()

    print("\n\n=== LOG_ENABLED = True (детальный лог, тест стагнации) ===")
    LOG_ENABLED = True
    demo_stagnation()
    LOG_ENABLED = False

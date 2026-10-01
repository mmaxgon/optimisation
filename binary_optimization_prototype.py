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
# Параметры алгоритма (см. раздел 3.2 "Инициализация")
# ============================================================
EPS = 1e-6                    # порог целочисленности ε
RHO_HAT = 0.1                 # ρ̂: ρ_0 = ρ̂ · max|c_i| (привязка к масштабу c)
RHO_0 = RHO_HAT * np.max(np.abs(c_min))   # начальный штрафной коэффициент ρ_0
GAMMA = 0.1                   # γ: нижняя граница веса штрафа ρ·γ
DELTA = 0.1                   # δ: ширина зоны неопределённости вокруг 1/2
BETA = 2.0                    # β: коэффициент увеличения штрафа
RHO_MAX = 1e6 * RHO_0         # потолок штрафа ρ_max
R_MAX = 5                     # R: макс. число возмущений за одно посещение Шага 2
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

# --- Параметры warm start ---
WARM_START = True             # включить warm start для LP
WARM_PRESOLVE = False          # presolve=False при warm start (убирает overhead)
WARM_METHOD = 'highs-ds'      # dual simplex для ре-решений
MILP_WARM_START = True        # добавлять отсечение по лучшему решению в MILP
OBJ_CUT_TOL = 1e-6            # допуск для отсечения по целевой

rng = np.random.default_rng(SEED)
STATS = {}


def reset_stats():
    STATS.clear()
    STATS.update({'lp': 0, 'milp': 0, 'perturb': 0, 'dive': 0, 'backtrack': 0,
                  'best_updates': 0, 'warm_lp': 0, 'milp_warm': 0})


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
# Warm start для LP: dual simplex + отключение presolve
# ============================================================
class WarmStartCache:
    """Кэш для warm start LP-решений.

    scipy.linprog не передаёт warm start (базис) в HiGHS напрямую.
    Используется комбинированный подход:
      - method='highs-ds' (двойственный симплекс) для ре-решений — естественно
        быстрее, когда меняется только целевая функция (как в penalty_step),
        т.к. текущий базис остаётся двойственно-допустимым;
      - presolve=False для ре-решений — убирает накладные расходы на
        препроцессинг, когда структура задачи уже известна;
      - для нового набора переменных (фиксация в Шаге 1) — полный решатель
        с presolve=True.
    """

    def __init__(self):
        self.enabled = WARM_START
        self.last_J_key = None
        self.last_x_J = None
        self.n_warm = 0

    def get_options(self, I, J, mod_c=None):
        """Возвращает (method, options) для linprog.

        Если множество J совпадает с предыдущим решением и изменена
        только целевая функция (mod_c задан) — идеальный сценарий
        для dual simplex без presolve.
        """
        J_key = frozenset(J) if J else frozenset()

        if (self.enabled and self.last_J_key is not None
                and self.last_J_key == J_key and mod_c is not None):
            self.n_warm += 1
            STATS['warm_lp'] += 1
            return WARM_METHOD, {'presolve': WARM_PRESOLVE}

        return 'highs', {'presolve': True}

    def update(self, x_J, J):
        """Обновляет кэш после успешного LP-решения."""
        if x_J is not None:
            self.last_J_key = frozenset(J) if J else frozenset()
            self.last_x_J = x_J.copy()


# Глобальный кэш
warm_cache = WarmStartCache()


# ============================================================
# Вспомогательные функции
# ============================================================
def solve_lp(I, J, x_fixed, mod_c=None):
    """LP: min c_min[J] @ x_J + mod_c @ x_J
    при A_ub[:,J] @ x_J <= b_ub - A_ub[:,I] @ x_I,  0 <= x <= 1.

    Warm start: при совпадении множества J с предыдущим решением
    и изменённой целевой функции используется dual simplex (highs-ds)
    с presolve=False.
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

    b = b_ub.copy()
    if len(I_arr) > 0:
        b = b - A_ub[:, I_arr] @ x_fixed[I_arr]

    STATS['lp'] += 1

    method, options = warm_cache.get_options(I, J, mod_c)
    res = linprog(c=obj, A_ub=A_ub[:, J_arr], b_ub=b,
                  bounds=(0, 1), method=method, options=options)
    if not res.success:
        return None, None

    x_full = x_fixed.copy()
    x_full[J_arr] = res.x

    warm_cache.update(res.x, set(J_arr))
    return x_full, float(res.fun)


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

    Warm start: при последовательных LP с одним и тем же множеством J
    (изменяется только целевая функция) используется dual simplex
    с presolve=False.

    Возвращает:
      'candidates' — появились почти бинарные переменные (st.x_bar обновлён,
                     ρ ← max(ρ_0, ρ/β)), далее Шаг 1 с меткой '2';
      'exhausted'  — штраф достиг ρ_max либо стагнация не снимается R
                     возмущениями, далее Шаг 2′ (ρ сбрасывается в ρ_0).
    """
    J_arr = np.array(sorted(st.J), dtype=int)
    x_bar = st.x_bar.copy()
    rho = st.rho
    r = 0                                    # счётчик возмущений в этом посещении
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
            # Решение сдвинулось: x̄ ← x, ρ ← βρ
            x_bar = x_new
            history.append(xn.copy())
            flip = None
            rho *= BETA
            if rho > RHO_MAX:
                break
        elif r < R_MAX:
            # Рост ρ ничего не изменит — возмущение: меняем знак d_i у
            # ⌈θ|J|⌉ самых неопределённых переменных (при равенстве — случайных)
            r += 1
            STATS['perturb'] += 1
            m = max(1, int(np.ceil(THETA * len(J_arr))))
            order = np.lexsort((rng.random(len(xb)), np.abs(slope)))
            flip = order[:m]
        else:
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
    global rng, best, warm_cache
    rng = np.random.default_rng(seed)
    reset_stats()
    best = BestSolution()
    warm_cache = WarmStartCache()
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
    print(f"  Warm start LP: {warm_cache.n_warm} раз использован dual simplex")
    return df


if __name__ == '__main__':
    run_all()

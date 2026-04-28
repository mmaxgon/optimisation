"""
schedule_workers.py — Иерархический решатель для оптимизации расписания кинотеатра.

Реализует трёхуровневую декомпозицию задачи:
  Уровень 1 (get_movie_hall_count):  сколько сеансов каждого фильма в каждом зале (SCIP)
  Уровень 2 (get_movie_hall_seq):    в каком порядке идут фильмы в каждом зале (эвристика)
  Уровень 3 (get_schedule):          в какое время начинается каждый сеанс (точный DP)

А также функции для генерации и поиска решений:
  sample_initial()    — генерация случайного начального решения (все 3 уровня)
  search_neighborhood — поиск нового решения в окрестности текущего
  get_best_sequence   — мульти-кандидатная обёртка над уровнями 2+3
"""

from time import time
import random
import numpy as np
import pyscipopt as scip
from schedule_data import *

############################################################################
# Вспомогательные векторы для работы с параметрами решений
############################################################################

# sigma — вектор масштабов шума для каждой активной пары (зал, фильм).
# Используется в search_neighborhood для генерации нормального шума:
#   p1_noisy = N(p1, (alpha^k) * sigma)
# Равен верхним границам hall_movie_max_shows, чтобы шум был пропорционален
# максимально возможному числу сеансов.
sigma = np.array([float(hall_movie_max_shows[h, m]) for h, m in active_pairs])

# x_lb — нижние границы переменных (всегда 0: сеансов может не быть).
x_lb = np.zeros(n_active, dtype=int)

# x_ub — верхние границы переменных (максимум сеансов для каждой пары).
x_ub = np.array([hall_movie_max_shows[h, m] for h, m in active_pairs], dtype=int)


def dict_to_vec(d):
    """
    Преобразует словарь {(зал, фильм): значение} в numpy-вектор.

    Порядок элементов строго соответствует active_pairs.
    Используется для перехода от словарного представления решения
    к векторному при генерации шума в search_neighborhood.
    """
    return np.array([d[h, m] for h, m in active_pairs], dtype=float)


def vec_to_full_dict(v):
    """
    Преобразует numpy-вектор обратно в полный словарь {(зал, фильм): значение}.

    Включает ВСЕ пары (зал, фильм), даже неактивные (для которых hall_movie_max_shows = 0).
    Значения округляются до целых и ограничиваются нулём снизу.
    """
    d = {(h, m): 0 for h in halls for m in movies}
    for i, (h, m) in enumerate(active_pairs):
        d[h, m] = max(0, int(round(v[i])))
    return d


#############################################################
# Уровень 1: сколько сеансов каждого фильма идёт в каждом зале
#############################################################

def get_movie_hall_count(movie_hall_count_rand):
    """
    Определяет, сколько сеансов каждого фильма проходит в каждом зале.

    Строит модель целочисленного линейного программирования (SCIP):
    - Переменные: movie_count[m], hall_count[h], movie_hall_count[h,m]
    - Ограничения:
        * sum(movie_hall_count[h,m] по h) = movie_count[m]  — баланс по фильмам
        * sum(movie_hall_count[h,m] по m) = hall_count[h]   — баланс по залам
        * sum((len[m]+1) * movie_hall_count[h,m] по m) <= T+1 — время в зале
    - Целевая: минимизация L1-отклонения от movie_hall_count_rand
      (чтобы новое распределение было близко к предыдущему)

    Args:
        movie_hall_count_rand: целевое/предыдущее распределение {(зал,фильм): кол-во}

    Returns:
        (movie_hall_count_res, solve_time):
            movie_hall_count_res — найденное распределение или None
            solve_time — время решения в секундах
    """
    model = scip.Model("Сколько сеансов каждого фильма идёт в каждом зале")

    # --- Переменные ---

    # movie_count[m] — общее число сеансов фильма m по всем залам.
    # Границы: [movies[m].min, movies[m].max] — фильм нужно показать хотя бы min раз.
    movie_count = {
        m: model.addVar(
            name=f"{m}_count",
            lb=movies[m].min,
            ub=movies[m].max,
            vtype="I"  # Целочисленная переменная
        ) for m in movies
    }

    # hall_count[h] — общее число сеансов в зале h.
    # Границы: [0, hall_max_shows[h]].
    hall_count = {
        h: model.addVar(
            name=f"{h}_count",
            lb=0,
            ub=hall_max_shows[h],
            vtype="I"
        ) for h in halls
    }

    # movie_hall_count[h,m] — число сеансов фильма m в зале h.
    # Границы: [0, hall_movie_max_shows[h,m]].
    # Для несовместимых пар (зал не поддерживает формат фильма) верхняя граница = 0.
    movie_hall_count = {
        (h, m): model.addVar(
            name=f"{h}_{m}_count",
            lb=0,
            ub=hall_movie_max_shows[h, m],
            vtype="I"
        ) for m in movies for h in halls
    }

    # --- Ограничения ---

    # Баланс по фильмам: сумма сеансов фильма по всем залам = movie_count
    for m in movies:
        model.addCons(sum(movie_hall_count[h, m] for h in halls) == movie_count[m])

    # Баланс по залам + ограничение по времени:
    # sum(movie_hall_count) = hall_count и
    # суммарная длительность всех сеансов (с 5-минутными перерывами) не превышает T.
    # (len + 1) — длительность фильма плюс один шаг перерыва.
    for h in halls:
        model.addCons(sum(movie_hall_count[h, m] for m in movies) == hall_count[h])
        model.addCons(sum((movies[m].len + 1) * movie_hall_count[h, m] for m in movies) <= T + 1)

    # --- Целевая функция: минимизация L1-отклонения ---

    # mu[h,m] — вспомогательная переменная, моделирующая |movie_hall_count - reference|.
    # Линейная релаксация: mu >= x - ref и mu >= ref - x  =>  mu >= |x - ref|
    mu = {(h, m): model.addVar(name=f"mu[{h}, {m}]", vtype="C") for m in movies for h in halls}
    for h in halls:
        for m in movies:
            model.addCons(movie_hall_count[h, m] - movie_hall_count_rand[h, m] <= mu[h, m])
            model.addCons(movie_hall_count_rand[h, m] - movie_hall_count[h, m] <= mu[h, m])
    # Минимизируем суммарное отклонение по всем парам
    model.setObjective(sum(mu[h, m] for h in halls for m in movies), sense='minimize')

    # --- Параметры и решение ---
    model.setParam("display/verblevel", 0)  # Отключаем вывод SCIP в консоль
    model.setParam('limits/time', 10)       # Лимит: 10 секунд на одну задачу

    start_time = time()
    model.optimize()
    sol = model.getBestSol()       # Лучшее найденное решение
    end_time = time()
    status = model.getStatus()     # Статус: "optimal", "gaplimit", "infeasible" и др.

    if status in ("optimal", "gaplimit"):
        # Решение найдено — извлекаем округлённые целочисленные значения
        movie_hall_count_res = {
            (h, m): int(round(sol[movie_hall_count[h, m]]))
            for h in halls for m in movies
        }
    else:
        movie_hall_count_res = None

    return (movie_hall_count_res, end_time - start_time)


#############################################################
# Уровень 2: Последовательность фильмов в зале
#############################################################

def get_movie_hall_seq(movie_hall_count, ref_schedule=None):
    """
    Генерирует последовательность (порядок) показа фильмов в каждом зале.

    Для каждого зала h:
    - Если есть ref_schedule, берёт его последовательность как базу и
      корректирует: удаляет лишние сеансы или добавляет недостающие.
    - Если ref_schedule нет, генерирует случайную перестановку фильмов.

    Порядок фильмов в зале определяет, какие времена начала им доступны:
    первый сеанс может начаться в любой момент, каждый следующий —
    только после окончания предыдущего + 1 шаг перерыва.

    Args:
        movie_hall_count: {(зал, фильм): кол-во} — распределение сеансов
        ref_schedule: ссылочное расписание {(зал, индекс): {movie, start}} или None

    Returns:
        (halls_show_count, res, solve_time, pref_starts):
            halls_show_count — {зал: общее число сеансов}
            res — {(зал, индекс): фильм} — последовательность
            pref_starts — {(зал, индекс): время} — предпочтительные времена начала
    """
    # Вычисляем общее число сеансов в каждом зале
    halls_show_count = {h: sum(movie_hall_count[h, m] for m in movies) for h in halls}

    start_time = time()
    res = {}          # {(зал, индекс): имя_фильма}
    pref_starts = {}  # {(зал, индекс): предпочитаемое_время}

    for h in halls:
        new_counts = {m: movie_hall_count[h, m] for m in movies}  # Целевое кол-во по фильмам

        if ref_schedule is not None:
            # --- Есть ссылочное расписание: инкрементальная модификация ---

            # Извлекаем показы данного зала из ссылочного расписания, сортируем по индексу
            ref_shows = sorted(
                [((hall, i), info) for (hall, i), info in ref_schedule.items() if hall == h],
                key=lambda x: x[0][1]
            )
            ref_items = [info["movie"] for _, info in ref_shows]  # Фильмы по порядку
            ref_times = [info["start"] for _, info in ref_shows]  # Их времена начала
            old_counts = {m: ref_items.count(m) for m in movies}  # Старое кол-во по фильмам

            items = list(ref_items)  # Текущая последовательность (мутируемая)
            times = list(ref_times)

            # Удаляем лишние сеансы: если фильм стал показываться реже,
            # случайно удаляем нужное число его вхождений
            for m in movies:
                diff = old_counts[m] - new_counts[m]
                if diff > 0:
                    for _ in range(diff):
                        indices = [idx for idx, mv in enumerate(items) if mv == m]
                        if indices:
                            rm_idx = random.choice(indices)  # Случайный выбор удаляемого
                            items.pop(rm_idx)
                            times.pop(rm_idx)

            # Добавляем недостающие сеансы: если фильм стал показываться чаще,
            # вставляем новые вхождения в случайные позиции
            for m in movies:
                diff = new_counts[m] - old_counts[m]
                if diff > 0:
                    for _ in range(diff):
                        pos = random.randint(0, len(items))
                        items.insert(pos, m)
                        times.insert(pos, None)  # Нет предпочтительного времени

            # Сохраняем предпочтительные времена для сеансов, унаследованных от ref
            for i, t in enumerate(times):
                if t is not None:
                    pref_starts[(h, i)] = t
        else:
            # --- Нет ссылочного расписания: случайная перестановка ---
            elements = {m: new_counts[m] for m in movies if new_counts[m] > 0}
            items = []
            for item, count in elements.items():
                items.extend([item] * count)  # Каждый фильм — count раз
            random.shuffle(items)  # Случайный порядок

        # Записываем итоговую последовательность для зала h
        for i in range(halls_show_count[h]):
            res[h, i] = items[i]

    end_time = time()
    return (halls_show_count, res, end_time - start_time, pref_starts)


def get_best_sequence(movie_hall_count, ref_schedule=None, n_candidates=3):
    """
    Пробует несколько вариантов последовательности и выбирает лучший по продажам.

    Для каждого кандидата:
      1. Генерирует последовательность через get_movie_hall_seq
         (первый кандидат использует ref_schedule, остальные — случайные)
      2. Решает расписание через get_schedule (DP, максимизация продаж)
      3. Оценивает через get_goal

    Возвращает кандидата с максимальным значением целевой функции.

    Args:
        movie_hall_count: {(зал, фильм): кол-во}
        ref_schedule: ссылочное расписание для первого кандидата или None
        n_candidates: число вариантов (3 по умолчанию)

    Returns:
        (halls_show_count, best_seq, best_schedule, best_f_val)
        или (None, None, None, -1), если все кандидаты неудачны
    """
    best_f = -1
    best_seq = None
    best_schedule = None
    best_halls_show_count = None

    for attempt in range(n_candidates):
        # Первый кандидат использует ref_schedule, остальные — случайные
        halls_show_count, seq, _, pref_starts = get_movie_hall_seq(
            movie_hall_count,
            ref_schedule if attempt == 0 else None
        )
        # Решаем расписание (DP максимизирует продажи для данной последовательности)
        schedule, _ = get_schedule(halls_show_count, seq)
        if schedule is not None:
            f_val = get_goal(schedule)
            if f_val > best_f:
                best_f = f_val
                best_seq = seq
                best_schedule = schedule
                best_halls_show_count = halls_show_count

    return (best_halls_show_count, best_seq, best_schedule, best_f)


#############################################################
# Уровень 3: Оптимальное время начала сеансов (точный DP)
#############################################################

def get_schedule(halls_show_count, movie_hall_seq, pref_starts=None):
    """
    Определяет оптимальное время начала каждого сеанса в каждом зале.

    Использует динамическое программирование (DP). Для каждого зала
    задача решается независимо, поскольку залы не взаимодействуют.

    Алгоритм для одного зала с n сеансами в фиксированном порядке:
    ---------------------------------------------------------------
    Обратный проход (i = n-1 .. 0):
      dp[i][t] = максимальные суммарные продажи для сеансов i..n-1,
                 если сеанс i начинается в момент t.
      suf[i][t] = max(dp[i][t'] для t' >= t) — суффиксный максимум.

    Переход:
      dp[i][t] = sales[movie_i][t] + suf[i+1][t + len_i + 1]
                                           ^^^^^^^^^^^^^^^^^^
                      следующий сеанс может начаться не раньше t + len + 1

    Прямой проход:
      Для каждого сеанса i выбираем t, максимизирующий dp[i][t],
      при условии t >= earliest (время окончания предыдущего сеанса + перерыв).

    Сложность: O(n * T) на зал, где n — число сеансов, T = 288.
    Это значительно быстрее CP-SAT с add_element, при этом даёт точное решение.

    Args:
        halls_show_count: {зал: число_сеансов}
        movie_hall_seq: {(зал, индекс): фильм} — последовательность
        pref_starts: не используется (оставлено для совместимости сигнатур)

    Returns:
        (schedule, solve_time):
            schedule — {(зал, индекс): {movie, start, end}} или None
            solve_time — время выполнения
    """
    NEG_INF = -100000  # Значение "минус бесконечность" для недопустимых состояний
    start_time = time()

    schedule = {}
    feasible = True

    for h in halls:
        n = halls_show_count[h]
        if n == 0:
            continue  # Пустой зал — пропускаем

        # Предварительно извлекаем данные для скорости
        show_movies = [movie_hall_seq[h, i] for i in range(n)]
        show_lens = [movies[m].len for m in show_movies]

        # Границы допустимых времён начала для каждого сеанса:
        #   ub[i] — верхняя: после i должны поместиться сеансы i+1..n-1 + перерывы
        #   lb[i] — нижняя: до i должны поместиться сеансы 0..i-1 + перерывы
        suffix_len = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_len[i] = suffix_len[i + 1] + show_lens[i]
        ub = [T - suffix_len[i] - (n - 1 - i) for i in range(n)]
        prefix_len = [0] * (n + 1)
        for i in range(n):
            prefix_len[i + 1] = prefix_len[i] + show_lens[i]
        lb = [prefix_len[i] + i for i in range(n)]

        # Быстрая проверка: все ли сеансы физически помещаются в T?
        # Минимальная суммарная длительность = sum(len) + (n-1) перерывов
        min_total = sum(show_lens) + (n - 1)
        if min_total > T:
            feasible = False
            break

        # --- Обратный проход: вычисляем dp и суффиксные максимумы ---

        # Массивы хранятся с offset: dp[i][t - lb[i]] для t в [lb[i], ub[i]]
        dp_arrays = [None] * n
        suf_arrays = [None] * n

        # База: последний сеанс (i = n-1)
        i_last = n - 1
        lb_last, ub_last = lb[i_last], ub[i_last]
        size_last = ub_last - lb_last + 1
        dp_last = [NEG_INF] * size_last
        for t in range(lb_last, ub_last + 1):
            dp_last[t - lb_last] = int(sales[show_movies[i_last]][t])
        dp_arrays[i_last] = dp_last

        # Суффиксный максимум для последнего сеанса
        suf = [NEG_INF] * (size_last + 1)
        for j in range(size_last - 1, -1, -1):
            suf[j] = max(suf[j + 1], dp_last[j])
        suf_arrays[i_last] = suf

        # Остальные сеансы: от предпоследнего к первому (i = n-2 .. 0)
        for i in range(n - 2, -1, -1):
            m = show_movies[i]
            length = show_lens[i]
            lb_i, ub_i = lb[i], ub[i]
            size_i = ub_i - lb_i + 1
            next_suf = suf_arrays[i + 1]
            next_lb = lb[i + 1]
            next_len = len(next_suf)
            dp_cur = [NEG_INF] * size_i
            for t in range(lb_i, ub_i + 1):
                next_t = t + length + 1
                idx = next_t - next_lb
                future = next_suf[idx] if 0 <= idx < next_len else NEG_INF
                dp_cur[t - lb_i] = int(sales[m][t]) + future
            dp_arrays[i] = dp_cur
            # Суффиксный максимум для этого сеанса
            suf = [NEG_INF] * (size_i + 1)
            for j in range(size_i - 1, -1, -1):
                suf[j] = max(suf[j + 1], dp_cur[j])
            suf_arrays[i] = suf

        # --- Прямой проход: восстанавливаем оптимальные времена начала ---

        earliest = 0
        for i in range(n):
            m = show_movies[i]
            length = show_lens[i]
            lo = max(earliest, lb[i])
            hi = ub[i]
            dp_arr = dp_arrays[i]
            off = lb[i]
            best_t = lo
            best_val = NEG_INF
            for t in range(lo, hi + 1):
                val = dp_arr[t - off]
                if val > best_val:
                    best_val = val
                    best_t = t
            if best_val <= NEG_INF:
                # Нет допустимого времени — расписание невозможно
                feasible = False
                break
            # Записываем результат: фильм m в зале h, сеанс i
            schedule[(h, i)] = {"movie": m, "start": best_t, "end": best_t + length}
            # Следующий сеанс может начаться только после перерыва
            earliest = best_t + length + 1

    end_time = time()

    if not feasible:
        return (None, end_time - start_time)
    return (schedule, end_time - start_time)


#########################################################################
# Функция цели
#########################################################################

def get_goal(schedule):
    """
    Вычисляет суммарные ожидаемые продажи для данного расписания.

    Для каждого сеанса берёт sales[movie][start_time] из матрицы продаж
    и суммирует по всем сеансам. Это значение нужно максимизировать.

    Args:
        schedule: {(зал, индекс): {movie, start, end}}

    Returns:
        int: суммарные продажи
    """
    return sum(sales[schedule[ix]["movie"]][schedule[ix]["start"]] for ix in schedule.keys())


#########################################################################
# Вспомогательные функции для метаэвристики
#########################################################################

def perturb_sequence(seq, halls_show_count, swap_prob):
    """
    Модифицирует последовательность фильмов случайными обменами соседей.

    Для каждого зала проходит по всем соседним парам сеансов и с вероятностью
    swap_prob меняет их местами. Это простейший оператор мутации порядка.

    Args:
        seq: {(зал, индекс): фильм}
        halls_show_count: {зал: число_сеансов}
        swap_prob: вероятность обмена пары (0..1)

    Returns:
        dict: новая модифицированная последовательность
    """
    new_seq = dict(seq)
    for h in halls:
        n = halls_show_count[h]
        for i in range(n - 1):
            if random.random() < swap_prob:
                new_seq[h, i], new_seq[h, i + 1] = new_seq[h, i + 1], new_seq[h, i]
    return new_seq


def compute_packed_starts(seq, halls_show_count):
    """
    Вычисляет времена начала "упаковкой влево": каждый сеанс начинается
    сразу после окончания предыдущего (без свободных окон).

    Используется для генерации начальных приближений, когда нет
    предпочтительных времён из предыдущего расписания.

    Args:
        seq: {(зал, индекс): фильм}
        halls_show_count: {зал: число_сеансов}

    Returns:
        {(зал, индекс): время_начала}
    """
    starts = {}
    for h in halls:
        t = 0  # Начинаем с начала дня (шаг 0)
        for i in range(halls_show_count[h]):
            starts[(h, i)] = t
            t += movies[seq[h, i]].len + 1  # Длина фильма + перерыв 5 мин
    return starts


def sample_initial():
    """
    Генерирует случайное начальное решение (все 3 уровня).

    Алгоритм:
      1. Случайный вектор числа сеансов для активных пар (зал, фильм)
      2. Уровень 1 (SCIP): корректировка распределения до допустимого
      3. Уровни 2+3: get_best_sequence пробует 3 варианта порядка
         и выбирает лучший (DP максимизирует продажи для каждого)

    Returns:
        (x_dict, schedule, f_val) — распределение, расписание, значение цели
        или None, если решение не найдено
    """
    # Генерируем случайный вектор: для каждой активной пары (зал, фильм)
    # выбираем число сеансов равномерно из [0, max]
    a_vec = np.array([np.random.randint(x_lb[i], x_ub[i] + 1) for i in range(n_active)])

    # Уровень 1: SCIP подгоняет распределение под ограничения
    x_dict, _ = get_movie_hall_count(vec_to_full_dict(a_vec))
    if x_dict is None:
        return None

    # Уровни 2+3: пробуем 3 случайных порядка, DP оптимизирует продажи
    _, _, schedule, f_val = get_best_sequence(x_dict, n_candidates=3)
    if schedule is None:
        return None

    return (x_dict, schedule, f_val)


def search_neighborhood(ref_solution, k, alpha, sigma_y):
    """
    Поиск нового решения в окрестности текущего (для метаэвристики).

    Уровень 1 (мутация распределения):
      - Текущее распределение x_ref преобразуется в вектор
      - К вектору добавляется нормальный шум с дисперсией (alpha^k) * sigma
      - alpha^k убывает с итерациями → шум уменьшается → сужение поиска
      - Зашумлённый вектор подаётся в SCIP для корректировки

    Уровни 2+3 (мульти-кандидатная оптимизация):
      - get_best_sequence пробует 3 варианта порядка
      - Первый вариант использует ref_schedule как базу
      - DP максимизирует продажи для каждого варианта
      - Выбирается лучший

    Args:
        ref_solution: (x_ref, schedule_ref, _) — текущее лучшее решение
        k: номер итерации (управляет масштабом шума)
        alpha: коэффициент затухания шума (0 < alpha < 1)
        sigma_y: базовый масштаб шума

    Returns:
        (x_new, schedule, f_val) или None
    """
    x_ref, schedule_ref, _ = ref_solution

    # --- Уровень 1: зашумление вектора movie_hall_count ---
    p1 = dict_to_vec(x_ref)
    # Нормальный шум с убывающей дисперсией: sigma * alpha^k
    p1_noisy = np.random.normal(p1, (alpha ** k) * sigma)
    # Округляем и ограничиваем до допустимых целых значений
    p1_noisy = np.clip(np.round(p1_noisy), x_lb, x_ub)
    # Решаем SCIP для получения допустимого распределения
    x_new, _ = get_movie_hall_count(vec_to_full_dict(p1_noisy))
    if x_new is None:
        return None

    # --- Уровни 2+3: мульти-кандидатная оптимизация ---
    _, _, schedule, f_val = get_best_sequence(
        x_new, ref_schedule=schedule_ref, n_candidates=3
    )
    if schedule is None:
        return None

    return (x_new, schedule, f_val)

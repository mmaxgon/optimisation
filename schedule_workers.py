from time import time
import random
import numpy as np
from ortools.sat.python import cp_model
import pyscipopt as scip
from schedule_data import *

sigma = np.array([float(hall_movie_max_shows[h, m]) for h, m in active_pairs])
x_lb = np.zeros(n_active, dtype=int)
x_ub = np.array([hall_movie_max_shows[h, m] for h, m in active_pairs], dtype=int)

def dict_to_vec(d):
    """Извлечь активные компоненты из полного словаря в вектор."""
    return np.array([d[h, m] for h, m in active_pairs], dtype=float)

def vec_to_full_dict(v):
    """Преобразовать вектор активных компонент в полный словарь."""
    d = {(h, m): 0 for h in halls for m in movies}
    for i, (h, m) in enumerate(active_pairs):
        d[h, m] = max(0, int(round(v[i])))
    return d

#############################################################
# Решение 1: сколько сеансов каждого фильма идёт в каждом зале
#############################################################

def get_movie_hall_count(movie_hall_count_rand):
    """
    Определяет количество сеансов каждого фильма в каждом зале.

    Решает задачу целочисленного программирования с использованием SCIP.
    Целевая функция минимизирует абсолютное отклонение от предыдущего распределения
    (movie_hall_count_rand), чтобы обеспечить плавные изменения при итерациях.

    Переменные:
        movie_count[m]      - общее число сеансов фильма m во всех залах
        hall_count[h]       - общее число сеансов в зале h
        movie_hall_count[h,m] - число сеансов фильма m в зале h

    Ограничения:
        - Сумма сеансов фильма по залам равна общему числу сеансов фильма
        - Сумма сеансов в зале по фильмам равна общему числу сеансов в зале
        - Общая продолжительность всех сеансов в зале (с учетом 5-минутных перерывов)
          не превышает длину периода T

    Параметры:
        movie_hall_count_rand (dict): Предыдущее/случайное распределение сеансов по парам (зал, фильм)

    Возвращает:
        tuple: (movie_hall_count_res, solve_time), где
            movie_hall_count_res (dict) - найденное распределение или None, если решения нет
            solve_time (float)          - время решения в секундах
    """
    model = scip.Model("Cколько сеансов каждого фильма идёт в каждом зале")

    movie_count = {
        m: model.addVar(
            name=f"{m}_count",
            lb=movies[m].min,
            ub=movies[m].max,
            vtype="I"
        ) for m in movies
    }

    hall_count = {
        h: model.addVar(
            name=f"{h}_count",
            lb=0,
            ub=hall_max_shows[h],
            vtype="I"
        ) for h in halls
    }

    movie_hall_count = {
        (h, m): model.addVar(
            name=f"{h}_{m}_count",
            lb=0,
            ub=hall_movie_max_shows[h, m],
            vtype="I"
        ) for m in movies for h in halls
    }


    for m in movies:
        model.addCons(sum(movie_hall_count[h, m] for h in halls) == movie_count[m])
    for h in halls:
        model.addCons(sum(movie_hall_count[h, m] for m in movies) == hall_count[h])
        model.addCons(sum((movies[m].len + 1) * movie_hall_count[h, m] for m in movies) <= T + 1)

    mu = {(h, m): model.addVar(name=f"mu[{h}, {m}]", vtype="C") for m in movies for h in halls}
    for h in halls:
        for m in movies:
            model.addCons(movie_hall_count[h, m] - movie_hall_count_rand[h, m] <= mu[h, m])
            model.addCons(movie_hall_count_rand[h, m] - movie_hall_count[h, m] <= mu[h, m])
    model.setObjective(sum(mu[h, m] for h in halls for m in movies), sense='minimize')

    model.setParam("display/verblevel", 0)
    model.setParam('limits/time', 10)

    start_time = time()
    model.optimize()
    sol = model.getBestSol()
    end_time = time()
    status = model.getStatus()

    if status in ("optimal", "gaplimit"):
        movie_hall_count_res = {(h, m): int(round(sol[movie_hall_count[h, m]])) for h in halls for m in movies}
    else:
        movie_hall_count_res = None

    return (movie_hall_count_res, end_time - start_time)


#############################################################
# Решение 2: Последовательность фильмов в зале
#############################################################

def get_movie_hall_seq(movie_hall_count, ref_schedule=None):
    """
    Генерирует последовательность показа фильмов в каждом зале.

    На основе заданного количества сеансов фильмов в залах (movie_hall_count),
    формирует порядок их показа. При наличии ссылочного расписания (ref_schedule)
    использует его для инициализации последовательности, затем корректирует количество
    сеансов фильмов (добавляет/удаляет) в соответствии с movie_hall_count.

    Если ref_schedule не задан, генерирует случайную перестановку.

    Сохраняет предпочтительные времена начала (pref_starts) из ref_schedule
    для уже существующих сеансов.

    Параметры:
        movie_hall_count (dict): Число сеансов { (зал, фильм): количество }.
        ref_schedule (dict or None): Ссылочное расписание в формате {(зал, индекс): {"movie": имя, "start": время}}.

    Возвращает:
        tuple: (halls_show_count, res, solve_time, pref_starts), где
            halls_show_count (dict) - общее число сеансов в каждом зале
            res (dict)             - последовательность фильмов {(зал, индекс): фильм}
            solve_time (float)      - время выполнения
            pref_starts (dict)      - предпочтительные времена начала {(зал, индекс): время}
    """
    halls_show_count = {h: sum(movie_hall_count[h, m] for m in movies) for h in halls}

    start_time = time()
    res = {}
    pref_starts = {}
    for h in halls:
        new_counts = {m: movie_hall_count[h, m] for m in movies}

        if ref_schedule is not None:
            ref_shows = sorted(
                [((hall, i), info) for (hall, i), info in ref_schedule.items() if hall == h],
                key=lambda x: x[0][1]
            )
            ref_items = [info["movie"] for _, info in ref_shows]
            ref_times = [info["start"] for _, info in ref_shows]
            old_counts = {m: ref_items.count(m) for m in movies}

            items = list(ref_items)
            times = list(ref_times)

            for m in movies:
                diff = old_counts[m] - new_counts[m]
                if diff > 0:
                    for _ in range(diff):
                        indices = [idx for idx, mv in enumerate(items) if mv == m]
                        if indices:
                            rm_idx = random.choice(indices)
                            items.pop(rm_idx)
                            times.pop(rm_idx)

            for m in movies:
                diff = new_counts[m] - old_counts[m]
                if diff > 0:
                    for _ in range(diff):
                        pos = random.randint(0, len(items))
                        items.insert(pos, m)
                        times.insert(pos, None)

            for i, t in enumerate(times):
                if t is not None:
                    pref_starts[(h, i)] = t
        else:
            elements = {m: new_counts[m] for m in movies if new_counts[m] > 0}
            items = []
            for item, count in elements.items():
                items.extend([item] * count)
            random.shuffle(items)

        for i in range(halls_show_count[h]):
            res[h, i] = items[i]
    end_time = time()

    return (halls_show_count, res, end_time - start_time, pref_starts)


#############################################################
# Решение 3: Время сеансов
#############################################################

def get_schedule(halls_show_count, movie_hall_seq, pref_starts=None):
    """
    Определяет точное время начала сеансов в каждом зале.

    Решает задачу расписания с помощью CP-SAT, учитывая заданную последовательность фильмов.
    Обеспечивает непересечение сеансов и соблюдение хронологии.

    При наличии pref_starts (предпочитаемых времён начала) формулирует задачу минимизации
    отклонений от этих времён.

    Переменные:
        start_movie_hall[h,i] - время начала i-го сеанса в зале h
        end_movie_hall[h,i]   - время окончания i-го сеанса в зале h

    Ограничения:
        - end = start + len
        - start[i] >= end[i-1] + 1 (зазор 5 минут)

    Параметры:
        halls_show_count (dict): Число сеансов в каждом зале {зал: количество}.
        movie_hall_seq (dict):   Последовательность фильмов {(зал, индекс): фильм}.
        pref_starts (dict or None): Предпочитаемые времена начала {(зал, индекс): время}.

    Возвращает:
        tuple: (schedule, solve_time), где
            schedule (dict) - расписание {(зал, индекс): {"movie": имя, "start": время, "end": время}} или None
            solve_time (float) - время решения
    """
    model = cp_model.CpModel()

    start_movie_hall = {
        (h, i): model.new_int_var(
            name=f"Начало сеанса {i} в зале {h} фильма {movie_hall_seq[h, i]}",
            lb=0,
            ub=T - movies[movie_hall_seq[h, i]].len
        )
        for (h, i) in movie_hall_seq.keys()
    }

    end_movie_hall = {
        (h, i): model.new_int_var(
            name=f"Конец сеанса {i} в зале {h} фильма {movie_hall_seq[h, i]}",
            lb=0,
            ub=T
        )
        for (h, i) in movie_hall_seq.keys()
    }

    for (h, i) in movie_hall_seq.keys():
        model.add(end_movie_hall[h, i] == start_movie_hall[h, i] + movies[movie_hall_seq[h, i]].len)

    for h in halls:
        for i in range(1, halls_show_count[h]):
            model.add(start_movie_hall[h, i] >= end_movie_hall[h, i - 1] + 1)

    if pref_starts:
        deviations = []
        for (h, i), pref in pref_starts.items():
            if (h, i) in start_movie_hall:
                dev = model.new_int_var(0, T, f"dev[{h},{i}]")
                model.add(dev >= start_movie_hall[h, i] - pref)
                model.add(dev >= pref - start_movie_hall[h, i])
                deviations.append(dev)
        if deviations:
            model.minimize(sum(deviations))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    start_time = time()
    status = solver.solve(model)
    end_time = time()

    res = {}
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for h in halls:
            for i in range(halls_show_count[h]):
                res[(h, i)] = {
                    "movie": movie_hall_seq[h, i],
                    "start": solver.value(start_movie_hall[h, i]),
                    "end": solver.value(end_movie_hall[h, i])
                }
    else:
        res = None

    return (res, end_time - start_time)


#########################################################################
# Функция цели
#########################################################################
def get_goal(schedule):
    return sum(sales[schedule[ix]["movie"]][schedule[ix]["start"]] for ix in schedule.keys())


def perturb_sequence(seq, halls_show_count, swap_prob):
    """Возмутить последовательность: случайные обмены соседних сеансов."""
    new_seq = dict(seq)
    for h in halls:
        n = halls_show_count[h]
        for i in range(n - 1):
            if random.random() < swap_prob:
                new_seq[h, i], new_seq[h, i + 1] = new_seq[h, i + 1], new_seq[h, i]
    return new_seq


def compute_packed_starts(seq, halls_show_count):
    """Вычислить start times при упаковке сеансов влево."""
    starts = {}
    for h in halls:
        t = 0
        for i in range(halls_show_count[h]):
            starts[(h, i)] = t
            t += movies[seq[h, i]].len + 1
    return starts


def sample_initial():
    """
    Начальная выборка: случайные параметры на каждом уровне.
    Возвращает (x_dict, schedule, f_val) или None.
    """
    a_vec = np.array([np.random.randint(x_lb[i], x_ub[i] + 1) for i in range(n_active)])
    x_dict, _ = get_movie_hall_count(vec_to_full_dict(a_vec))
    if x_dict is None:
        return None

    halls_show_count, seq, _, _ = get_movie_hall_seq(x_dict)
    schedule, _ = get_schedule(halls_show_count, seq)
    if schedule is None:
        return None

    f_val = get_goal(schedule)
    return (x_dict, schedule, f_val)


def search_neighborhood(ref_solution, k, alpha, sigma_y):
    """
    Иерархический поиск в окрестности хорошего решения.
    """
    x_ref, schedule_ref, _ = ref_solution

    # Уровень 1: movie_hall_count
    p1 = dict_to_vec(x_ref)
    p1_noisy = np.random.normal(p1, (alpha ** k) * sigma)
    p1_noisy = np.clip(np.round(p1_noisy), x_lb, x_ub)
    x_new, _ = get_movie_hall_count(vec_to_full_dict(p1_noisy))
    if x_new is None:
        return None

    # Уровень 2: последовательность
    halls_show_count, seq, _, pref_starts = get_movie_hall_seq(x_new, schedule_ref)
    swap_prob = (alpha ** k) * 0.5
    seq = perturb_sequence(seq, halls_show_count, swap_prob)

    # Уровень 3: времена начала
    if swap_prob > 0:
        pref_starts = compute_packed_starts(seq, halls_show_count)
    noise_y = (alpha ** k) * sigma_y
    for (h, i) in seq:
        if (h, i) in pref_starts:
            pref_starts[(h, i)] = int(np.clip(
                np.round(np.random.normal(pref_starts[(h, i)], noise_y)),
                0, T - 1
            ))
        else:
            pref_starts[(h, i)] = np.random.randint(0, T)
    schedule, _ = get_schedule(halls_show_count, seq, pref_starts if pref_starts else None)
    if schedule is None:
        return None

    f_val = get_goal(schedule)
    return (x_new, schedule, f_val)

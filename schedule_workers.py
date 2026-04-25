from time import time
import random
import numpy as np
from ortools.sat.python import cp_model
import pyscipopt as scip
from schedule_data import *

# Векторы и матрицы для быстрого доступа и операций
# sigma: максимальное возможное число сеансов для каждой пары (зал, фильм)
# x_lb: нижняя граница (0) для каждой переменной
# x_ub: верхняя граница (максимальное число сеансов) для каждой переменной
sigma = np.array([float(hall_movie_max_shows[h, m]) for h, m in active_pairs])
x_lb = np.zeros(n_active, dtype=int)
x_ub = np.array([hall_movie_max_shows[h, m] for h, m in active_pairs], dtype=int)

def dict_to_vec(d):
    """
    Преобразует словарь с ключами (зал, фильм) в одномерный numpy массив.
    Используется для эффективного хранения и обработки параметров решений.
    """
    return np.array([d[h, m] for h, m in active_pairs], dtype=float)

def vec_to_full_dict(v):
    """
    Преобразует одномерный numpy массив обратно в полный словарь.
    Отображает значения из вектора на все возможные пары (зал, фильм),
    даже если для них hall_movie_max_shows равно 0.
    """
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
    # Создаём модель SCIP для оптимизации
    model = scip.Model("Cколько сеансов каждого фильма идёт в каждом зале")

    # Добавляем переменные для общего числа сеансов каждого фильма
    movie_count = {
        m: model.addVar(
            name=f"{m}_count",
            lb=movies[m].min,
            ub=movies[m].max,
            vtype="I"  # Целочисленная переменная
        ) for m in movies
    }

    # Добавляем переменные для общего числа сеансов в каждом зале
    hall_count = {
        h: model.addVar(
            name=f"{h}_count",
            lb=0,
            ub=hall_max_shows[h],
            vtype="I"  # Целочисленная переменная
        ) for h in halls
    }

    # Добавляем переменные для числа сеансов каждого фильма в каждом зале
    movie_hall_count = {
        (h, m): model.addVar(
            name=f"{h}_{m}_count",
            lb=0,
            ub=hall_movie_max_shows[h, m],
            vtype="I"  # Целочисленная переменная
        ) for m in movies for h in halls
    }

    # Ограничение: Сумма сеансов фильма по залам должна равняться общему числу сеансов фильма
    for m in movies:
        model.addCons(sum(movie_hall_count[h, m] for h in halls) == movie_count[m])
    
    # Ограничение: Сумма сеансов в зале по фильмам должна равняться общему числу сеансов в зале
    # Ограничение: Общая продолжительность всех сеансов в зале (с перерывами) <= T
    for h in halls:
        model.addCons(sum(movie_hall_count[h, m] for m in movies) == hall_count[h])
        model.addCons(sum((movies[m].len + 1) * movie_hall_count[h, m] for m in movies) <= T + 1)

    # Вспомогательные переменные для минимизации абсолютного отклонения (L1)
    mu = {(h, m): model.addVar(name=f"mu[{h}, {m}]", vtype="C") for m in movies for h in halls}
    # Ограничения, определяющие mu как |movie_hall_count - movie_hall_count_rand|
    for h in halls:
        for m in movies:
            model.addCons(movie_hall_count[h, m] - movie_hall_count_rand[h, m] <= mu[h, m])
            model.addCons(movie_hall_count_rand[h, m] - movie_hall_count[h, m] <= mu[h, m])
    # Целевая функция: минимизация суммарного абсолютного отклонения
    model.setObjective(sum(mu[h, m] for h in halls for m in movies), sense='minimize')

    # Параметры решателя
    model.setParam("display/verblevel", 0)  # Уровень вывода в консоль (0 - ничего)
    model.setParam('limits/time', 10)       # Лимит времени на решение в секундах

    # Запускаем оптимизацию
    start_time = time()
    model.optimize()
    sol = model.getBestSol()  # Получаем лучшее найденное решение
    end_time = time()
    status = model.getStatus() # Получаем статус решения

    # Обработка результата
    if status in ("optimal", "gaplimit"):
        # Если решение найдено (оптимальное или с хорошим gap'ом), извлекаем его
        movie_hall_count_res = {(h, m): int(round(sol[movie_hall_count[h, m]])) for h in halls for m in movies}
    else:
        # Если решение не найдено, возвращаем None
        movie_hall_count_res = None

    # Возвращаем результат и время выполнения
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
    # Вычисляем общее число сеансов для каждого зала
    halls_show_count = {h: sum(movie_hall_count[h, m] for m in movies) for h in halls}

    start_time = time()
    res = {}          # Словарь для хранения последовательности фильмов
    pref_starts = {}  # Словарь для хранения предпочтительных времён начала
    
    # Генерируем последовательность для каждого зала
    for h in halls:
        # Получаем целевое количество сеансов каждого фильма в зале h
        new_counts = {m: movie_hall_count[h, m] for m in movies}

        # Если есть ссылочное расписание, используем его как базу
        if ref_schedule is not None:
            # Фильтруем и сортируем показы для текущего зала h
            ref_shows = sorted(
                [((hall, i), info) for (hall, i), info in ref_schedule.items() if hall == h],
                key=lambda x: x[0][1]
            )
            # Извлекаем имена фильмов и времена начала из ссылочного расписания
            ref_items = [info["movie"] for _, info in ref_shows]
            ref_times = [info["start"] for _, info in ref_shows]
            # Считаем, сколько раз каждый фильм уже показывался
            old_counts = {m: ref_items.count(m) for m in movies}

            # Начинаем с существующей последовательности
            items = list(ref_items)
            times = list(ref_times)

            # Если фильма стало меньше, удаляем лишние вхождения
            for m in movies:
                diff = old_counts[m] - new_counts[m]
                if diff > 0:
                    for _ in range(diff):
                        indices = [idx for idx, mv in enumerate(items) if mv == m]
                        if indices:
                            rm_idx = random.choice(indices)
                            items.pop(rm_idx)
                            times.pop(rm_idx)

            # Если фильмов стало больше, добавляем новые в случайные позиции
            for m in movies:
                diff = new_counts[m] - old_counts[m]
                if diff > 0:
                    for _ in range(diff):
                        pos = random.randint(0, len(items))
                        items.insert(pos, m)
                        times.insert(pos, None)

            # Сохраняем предпочтительные времена начала для сеансов, которые остались на своих местах
            for i, t in enumerate(times):
                if t is not None:
                    pref_starts[(h, i)] = t
        else:
            # Если нет ссылочного расписания, генерируем случайную последовательность
            # Собираем элементы и их количество
            elements = {m: new_counts[m] for m in movies if new_counts[m] > 0}
            items = []
            for item, count in elements.items():
                items.extend([item] * count)
            random.shuffle(items)

        # Записываем итоговую последовательность для зала h
        for i in range(halls_show_count[h]):
            res[h, i] = items[i]
    
    end_time = time()

    # Возвращаем результаты
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
    # Создаём модель CP-SAT
    model = cp_model.CpModel()

    # Переменные для времени начала каждого сеанса
    start_movie_hall = {
        (h, i): model.new_int_var(
            name=f"Начало сеанса {i} в зале {h} фильма {movie_hall_seq[h, i]}",
            lb=0,
            ub=T - movies[movie_hall_seq[h, i]].len
        )
        for (h, i) in movie_hall_seq.keys()
    }

    # Переменные для времени окончания каждого сеанса
    end_movie_hall = {
        (h, i): model.new_int_var(
            name=f"Конец сеанса {i} в зале {h} фильма {movie_hall_seq[h, i]}",
            lb=0,
            ub=T
        )
        for (h, i) in movie_hall_seq.keys()
    }

    # Ограничение: Время окончания = Время начала + Длительность фильма
    for (h, i) in movie_hall_seq.keys():
        model.add(end_movie_hall[h, i] == start_movie_hall[h, i] + movies[movie_hall_seq[h, i]].len)

    # Ограничение: Следующий сеанс начинается после окончания предыдущего + перерыв
    for h in halls:
        for i in range(1, halls_show_count[h]):
            model.add(start_movie_hall[h, i] >= end_movie_hall[h, i - 1] + 1)

    # Если заданы предпочтительные времена начала, добавляем задачу минимизации отклонений
    if pref_starts:
        deviations = []
        for (h, i), pref in pref_starts.items():
            if (h, i) in start_movie_hall:
                # Вспомогательная переменная для отклонения
                dev = model.new_int_var(0, T, f"dev[{h},{i}]")
                # Ограничения, определяющие dev как |start - pref|
                model.add(dev >= start_movie_hall[h, i] - pref)
                model.add(dev >= pref - start_movie_hall[h, i])
                deviations.append(dev)
        # Целевая функция: минимизация суммы отклонений
        if deviations:
            model.minimize(sum(deviations))


    # Создаём и настраиваем решатель
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    
    # Запускаем решение
    start_time = time()
    status = solver.solve(model)
    end_time = time()

    # Обработка результата
    res = {}
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Если решение найдено, извлекаем его
        for h in halls:
            for i in range(halls_show_count[h]):
                res[(h, i)] = {
                    "movie": movie_hall_seq[h, i],
                    "start": solver.value(start_movie_hall[h, i]),
                    "end": solver.value(end_movie_hall[h, i])
                }
    else:
        # Если решение не найдено, возвращаем None
        res = None

    # Возвращаем результат и время выполнения
    return (res, end_time - start_time)


#########################################################################
# Функция цели
#########################################################################
def get_goal(schedule):
    """
    Вычисляет суммарные ожидаемые продажи для данного расписания.

    Сумма значений из матрицы sales для каждого сеанса.
    """
    return sum(sales[schedule[ix]["movie"]][schedule[ix]["start"]] for ix in schedule.keys())


def perturb_sequence(seq, halls_show_count, swap_prob):
    """
    Модифицирует последовательность фильмов путем случайного обмена
    соседних сеансов с заданной вероятностью.

    Параметры:
        seq (dict): Последовательность фильмов {(зал, индекс): фильм}.
        halls_show_count (dict): Число сеансов в каждом зале.
        swap_prob (float): Вероятность обмена.

    Возвращает:
        new_seq (dict): Новая последовательность.
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
    Вычисляет времена начала сеансов, "упаковывая" их влево,
    то есть сразу после окончания предыдущего.

    Параметры:
        seq (dict): Последовательность фильмов.
        halls_show_count (dict): Число сеансов в каждом зале.

    Возвращает:
        starts (dict): Времена начала {(зал, индекс): время}.
    """
    starts = {}
    for h in halls:
        t = 0 # Начинаем с нулевого времени
        for i in range(halls_show_count[h]):
            starts[(h, i)] = t
            t += movies[seq[h, i]].len + 1 # Добавляем длину фильма и перерыв
    return starts


def sample_initial():
    """
    Генерирует случайное начальное решение.
    Создает случайные параметры для первого уровня и последовательно
    решает вторую и третью подзадачи.

    Возвращает:
        tuple: (x_dict, schedule, f_val) или None, если на каком-то этапе решение не найдено.
    """
    # Генерируем случайный вектор для числа сеансов
    a_vec = np.array([np.random.randint(x_lb[i], x_ub[i] + 1) for i in range(n_active)])
    # Решаем первую подзадачу
    x_dict, _ = get_movie_hall_count(vec_to_full_dict(a_vec))
    if x_dict is None:
        return None

    # Решаем вторую подзадачу
    halls_show_count, seq, _, _ = get_movie_hall_seq(x_dict)
    # Решаем третью подзадачу
    schedule, _ = get_schedule(halls_show_count, seq)
    if schedule is None:
        return None

    # Вычисляем значение функции цели
    f_val = get_goal(schedule)
    return (x_dict, schedule, f_val)


def search_neighborhood(ref_solution, k, alpha, sigma_y):
    """
    Иерархический поиск в окрестности хорошего решения.
    Для каждого уровня решения создает "шумные" параметры и решает подзадачу.

    Параметры:
        ref_solution (tuple): Референсное решение (x_ref, schedule_ref, _).
        k (int): Номер итерации (для управления масштабом шума).
        alpha (float): Коэффициент сжатия окрестности.
        sigma_y (float): Базовый масштаб шума для времени начала.

    Возвращает:
        tuple: (x_new, schedule, f_val) или None.
    """
    x_ref, schedule_ref, _ = ref_solution

    # Уровень 1: Генерируем новые параметры movie_hall_count
    p1 = dict_to_vec(x_ref)
    p1_noisy = np.random.normal(p1, (alpha ** k) * sigma)
    p1_noisy = np.clip(np.round(p1_noisy), x_lb, x_ub)
    x_new, _ = get_movie_hall_count(vec_to_full_dict(p1_noisy))
    if x_new is None:
        return None

    # Уровень 2: Генерируем новую последовательность
    halls_show_count, seq, _, pref_starts = get_movie_hall_seq(x_new, schedule_ref)
    swap_prob = (alpha ** k) * 0.5
    seq = perturb_sequence(seq, halls_show_count, swap_prob)

    # Уровень 3: Генерируем новые предпочтительные времена начала
    if swap_prob > 0:
        # Если была перестановка, используем упаковку влево
        pref_starts = compute_packed_starts(seq, halls_show_count)
    noise_y = (alpha ** k) * sigma_y
    for (h, i) in seq:
        if (h, i) in pref_starts:
            # Добавляем нормальный шум к существующему времени
            pref_starts[(h, i)] = int(np.clip(
                np.round(np.random.normal(pref_starts[(h, i)], noise_y)),
                0, T - 1
            ))
        else:
            # Для нового сеанса - случайное время
            pref_starts[(h, i)] = np.random.randint(0, T)
    
    # Решаем третью подзадачу
    schedule, _ = get_schedule(halls_show_count, seq, pref_starts if pref_starts else None)
    if schedule is None:
        return None

    # Вычисляем значение функции цели
    f_val = get_goal(schedule)
    return (x_new, schedule, f_val)

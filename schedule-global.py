from time import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import random
import numpy as np
from ortools.sat.python import cp_model
from schedule_data import *
from schedule_workers import sample_initial, search_neighborhood

"""
Глобальная оптимизация.
Разделена на 3 этапа принятия решений.
Результаты каждого этапа верхнего уровня служат ограничениями для следующего.
Идея:
1. Создаётся множество допустимых решений со случаным распределением числа сеансов фильмов по залам.
2. Из множества допустимых решений выбирается 50% лучших. Остальные 50% создаются "в окрестности" лучших.
Окрестность решения получается расчётом векторов, описывающих решения и служащих параметрами для нахождения решений на каждом уровне.
Т.е. решение -> вектор параметров шага 1, вектор параметров шага 2, вектор параметров шага 3.
Каждый вектор слегка зашумляется и подаётся на вход в решатель каждого уровня для поиска близких решений.
Идея: 50% новых решений ищутся в окрестности 50% лучших.
"""

# Параметры алгоритма
N = 20                    # размер популяции
alpha = 0.95               # коэффициент сжатия окрестности поиска
max_iter = 1000             # максимальное число итераций
stagnation_limit = 100      # останов при стагнации (итераций без улучшения)
SEED = 42

N_keep = N // 2  # число лучших точек, сохраняемых на каждой итерации
sigma_y = 3.0  # масштаб шума для предпочтительных времён начала (итерации)


def refine_with_cpsat(schedule, time_limit=600):
    """
    Уточняет найденное метаэвристикой расписание с помощью полной модели CP-SAT.

    Данный метод строит детальную модель ограничений и целевой функции с использованием
    Google OR-Tools CP-SAT решателя. Использует найденное расписание как "подсказку"
    (warm start), что может ускорить поиск оптимального или близкого к оптимальному решения.

    Args:
        schedule (dict): Расписание, найденное метаэвристикой. Ключи — пары (зал, индекс),
                         значения — словари с полями 'movie' (название фильма) и 'start' (время начала).
        time_limit (int, optional): Максимальное время решения в секундах. По умолчанию 600.

    Returns:
        tuple: Кортеж из четырёх элементов:
            - solver (cp_model.CpSolver): Решатель после выполнения.
            - status (int): Статус результата (например, OPTIMAL, FEASIBLE).
            - x (dict): Словарь булевых переменных модели, соответствующих началу сеансов.
            - dt (float): Время выполнения решения в секундах.
    """
    # Создаём модель CP-SAT
    model = cp_model.CpModel()

    # --- Переменные ---
    # Переменная для общего числа сеансов каждого фильма
    movie_count_var = {
        m: model.new_int_var(name=f"{m}_count", lb=movies[m].min, ub=movies[m].max)
        for m in movies
    }
    # Переменная для общего числа сеансов в каждом зале
    hall_count_var = {
        h: model.new_int_var(name=f"{h}_count", lb=0, ub=hall_max_shows[h])
        for h in halls
    }
    # Переменная для числа сеансов каждого фильма в каждом зале
    movie_hall_count_var = {
        (h, m): model.new_int_var(name=f"{h}_{m}_count", lb=0, ub=hall_movie_max_shows[h, m])
        for m in movies for h in halls
    }

    # Связываем переменные: sum(movie_hall_count) = movie_count
    for m in movies:
        model.add(sum(movie_hall_count_var[h, m] for h in halls) == movie_count_var[m])
    # Связываем переменные: sum(movie_hall_count) = hall_count
    for h in halls:
        model.add(sum(movie_hall_count_var[h, m] for m in movies) == hall_count_var[h])

    # Переменная для каждого возможного 5-минутного шага начала сеанса
    x = {
        (h, m, t): model.new_bool_var(name=f"x[{h}, {m}, {t}]")
        for h in halls for m in hall_movies[h] for t in valid_starts[m]
    }
    # Интервал сеанса (начало, длительность, конец)
    shows = {
        (h, m, t): model.new_optional_interval_var(
            start=t, size=movies[m].len + 1, end=t + movies[m].len + 1,
            is_present=x[h, m, t], name=f"show[{h}, {m}, {t}]"
        ) for (h, m, t) in x.keys()
    }

    # Связываем переменные: sum(x) = movie_hall_count
    for h in halls:
        for m in movies:
            if h in movie_halls[m]:
                model.add(sum(x[h, m, t] for t in valid_starts[m]) == movie_hall_count_var[h, m])
            else:
                model.add(movie_hall_count_var[h, m] == 0)

    # Ограничение: в каждый момент времени в зале начинается не больше одного сеанса
    for h in halls:
        for t in period:
            starts_at_t = [x[h, m, t] for m in hall_movies[h] if (h, m, t) in x]
            if starts_at_t:
                model.add(sum(starts_at_t) <= 1)

    # Ограничение: сеансы в зале не пересекаются
    for h in halls:
        hall_shows = [shows[h, m, t] for m in hall_movies[h] for t in valid_starts[m]]
        model.add_no_overlap(hall_shows)

    # Целевая функция: максимизация суммарных продаж
    model.maximize(sum(sales[m][t] * x[h, m, t] for (h, m, t) in x.keys()))

    # --- warm start: hint из расписания метаэвристики ---
    # Создаём множество активных сеансов из референсного расписания
    active_shows = set()
    for (h, i), info in schedule.items():
        active_shows.add((h, info["movie"], info["start"]))

    # Добавляем hints для булевых переменных x
    for (h, m, t) in x.keys():
        model.add_hint(x[h, m, t], 1 if (h, m, t) in active_shows else 0)

    # Добавляем hints для переменных уровня 1 (movie_count, hall_count и т.д.)
    hint_mhc = {(h, m): 0 for h in halls for m in movies}
    for (h, i), info in schedule.items():
        hint_mhc[h, info["movie"]] += 1
    for m in movies:
        model.add_hint(movie_count_var[m], sum(hint_mhc[h, m] for h in halls))
    for h in halls:
        model.add_hint(hall_count_var[h], sum(hint_mhc[h, m] for m in movies))
    for h in halls:
        for m in movies:
            model.add_hint(movie_hall_count_var[h, m], hint_mhc[h, m])

    # --- Решение ---
    # Создаём и настраиваем решатель
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    # Запускаем решение
    start = time()
    status = solver.solve(model)
    dt = time() - start

    # Возвращаем решатель, статус, переменные и время
    return solver, status, x, dt


def crossover_parents(params1, params2, alpha=0.5):
    """
    Выполняет арифметическое скрещивание двух родительских решений.

    Каждое решение представлено в виде словаря с количеством сеансов фильмов по залам.
    Результат — новое решение, полученное как взвешенное среднее родителей с округлением
    и проверкой границ.

    Args:
        params1 (dict): Первый родитель, словарь вида {(зал, фильм): количество}.
        params2 (dict): Второй родитель, аналогично.
        alpha (float, optional): Вес первого родителя в смешивании. По умолчанию 0.5.

    Returns:
        dict: Новое решение — потомок, с неотрицательными целыми значениями.
    """
    child_params = {}
    # Перебираем все возможные пары (зал, фильм)
    for key in params1.keys():
        val1 = params1[key]
        val2 = params2[key] if key in params2 else 0
        
        # Арифметическое скрещивание: взвешенное среднее
        mixed = alpha * val1 + (1 - alpha) * val2
        # Округление до ближайшего целого
        count = int(round(mixed))
        
        # Гарантируем неотрицательность
        child_params[key] = max(0, count)
    
    return child_params


def local_search_step(solution, max_perturb=2):
    """
    Выполняет локальный поиск вокруг заданного решения.

    Является заглушкой для будущих улучшений. Пока использует функцию
    `search_neighborhood` с малым уровнем шума для незначительных изменений.

    Args:
        solution (tuple): Текущее решение в виде (params, schedule_dict, score).
        max_perturb (int, optional): Максимальное число изменений (не используется напрямую).

    Returns:
        tuple: Новое решение после локального поиска.
    """
    schedule_data, schedule_dict, score = solution
    # Малый шум — только локальные изменения
    return search_neighborhood(solution, iter_num=0, alpha=1.0, sigma_y=0.5)


def main():
    """
    Основная функция запуска глобальной оптимизации расписания сеансов.

    Алгоритм работает в два этапа:
    1. Эволюционный поиск с использованием популяции решений, мутаций, скрещивания и параллелизации.
    2. Уточнение лучшего найденного решения с помощью CP-SAT решателя с warm start.

    Процесс:
    - Генерация начальной популяции допустимых решений.
    - Итеративное улучшение: отбор лучших, мутация, скрещивание, иммиграция.
    - Остановка при достижении лимита итераций или стагнации.
    - Финальное уточнение через CP-SAT.

    Печатает промежуточную и финальную статистику.
    """
    # Устанавливаем детерминированность для воспроизводимости
    np.random.seed(SEED)
    random.seed(SEED)

    # Определяем число рабочих процессов для параллельных вычислений
    n_workers = min(os.cpu_count() or 1, N)

    # --- Шаги 2-6: начальное исследование (параллельно) ---
    print("=" * 60)
    print(f"Алгоритм: N={N}, alpha={alpha}, max_iter={max_iter}, workers={n_workers}")
    print("=" * 60)

    # Запоминаем время начала
    t_total = time()

    # Используем пул процессов для параллельного запуска
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Начальная выборка: батчи по n_workers*3 задач
        population = []
        max_attempts = N * 5
        batch_size = n_workers * 3
        submitted = 0
        while len(population) < N and submitted < max_attempts:
            n_submit = min(batch_size, max_attempts - submitted)
            # Запускаем N задач sample_initial в параллель
            futures = [executor.submit(sample_initial) for _ in range(n_submit)]
            submitted += n_submit
            # Собираем результаты по мере их завершения
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception:
                    continue
                if result is not None:
                    population.append(result)
                # Если набрали достаточно, выходим
                if len(population) >= N:
                    break

        # Сортируем популяцию по убыванию значения целевой функции (f_val)
        population.sort(key=lambda r: r[2], reverse=True)  # maximize
        # Отбираем лучшие N_keep решений
        n_keep = min(N_keep, len(population))
        best = list(population[:n_keep])
        # История всех оценённых решений
        history = list(population)

        print(f"Начальная выборка: {len(population)} допустимых точек")
        print(f"Лучшее f = {best[0][2]}")

        # --- Шаги 7-11: итерации (параллельный поиск в окрестности + скрещивание) ---
        # Запоминаем лучшее значение цели
        prev_best_f = best[0][2]
        # Счётчик итераций без улучшения
        stagnation_count = 0

        # Основной цикл оптимизации
        for k in range(1, max_iter + 1):
            # Список задач для параллельного выполнения
            futures = []

            # 1. Мутация: исследуем окрестность лучших решений (как раньше)
            # Генерируем новые точки вокруг лучших решений
            for j in range(n_keep):
                futures.append(
                    executor.submit(search_neighborhood, best[j], k, alpha, sigma_y)
                )

            # 2. СКРЕЩИВАНИЕ: создаём новых кандидатов из пар лучших
            n_crossover = N_keep // 3  # ~30% от популяции — дети
            # Создаём потомков от пар лучших родителей
            if len(best) >= 2:
                for _ in range(n_crossover):
                    # Выбираем двух родителей (турнир или случайно)
                    p1, p2 = random.sample(best, 2)
                    # Извлекаем параметры movie_hall_count (первый элемент кортежа)
                    try:
                        child_params = crossover_parents(p1[0], p2[0], alpha=0.5)
                        # Создаём "фиктивное" решение: (params, schedule=None, score=0)
                        dummy_solution = (child_params, None, 0.0)
                        # Запускаем search_neighborhood с этим решением
                        # Меньше шума — ребёнок уже "хорош"
                        futures.append(
                            executor.submit(search_neighborhood, dummy_solution, k, alpha, sigma_y * 0.7)
                        )
                    except Exception as e:
                        # На случай ошибки — просто пропускаем
                        continue

            # 3. Редкая иммиграция (одно новое случайное решение каждые 10 итераций)
            # Вводим одно полностью новое случайное решение
            if k % 10 == 0:
                futures.append(executor.submit(sample_initial))

            # 4. Сбор результатов
            # Собираем все результаты из параллельных задач
            new_points = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        new_points.append(result)
                except Exception:
                    continue

            # 5. Объединение и отбор лучших
            # Объединяем лучшие решения прошлой итерации с новыми точками
            combined = best + new_points
            # Сортируем по убыванию значения целевой функции
            combined.sort(key=lambda r: r[2], reverse=True)
            # Отбираем снова N_keep лучших
            best = combined[:n_keep]
            # Добавляем новые точки в историю
            history.extend(new_points)

            # 6. Проверка улучшения
            current_best_f = best[0][2]
            # Если найдено лучшее решение, сбрасываем счётчик стагнации
            if current_best_f > prev_best_f:
                stagnation_count = 0
            else:
                stagnation_count += 1
            prev_best_f = current_best_f

            # Выводим статистику итерации
            print(f"Итерация {k:3d}: f = {current_best_f:6.1f} | "
                  f"новых = {len(new_points):2d} | стагнация = {stagnation_count} | "
                  f"t = {time() - t_total:.1f}s")

            # Проверка условия остановки по стагнации
            if stagnation_count >= stagnation_limit:
                print(f"Остановка: стагнация {stagnation_limit} итераций")
                break

    # --- Результат метаэвристики ---
    # Финальное значение целевой функции
    f_meta = best[0][2]
    # Общее время работы алгоритма
    t_meta = time() - t_total
    # Вывод результатов
    print("\n" + "=" * 60)
    print(f"МЕТАЭВРИСТИКА: f = {f_meta}, время = {t_meta:.1f}s")
    print(f"Всего вычислено точек: {len(history)}")

    # --- Уточнение через CP-SAT с warm start ---
    print("\n" + "=" * 60)
    print("Уточнение через CP-SAT (warm start)...")
    solver_cpsat, status_cpsat, x_cpsat, dt_cpsat = refine_with_cpsat(best[0][1], time_limit=600)

    print("\n" + "=" * 60)
    if status_cpsat == cp_model.OPTIMAL or status_cpsat == cp_model.FEASIBLE:
        f_cpsat = int(solver_cpsat.objective_value)
        delta = f_cpsat - f_meta
        print(f"CP-SAT:       f = {f_cpsat}, статус = {solver_cpsat.status_name(status_cpsat)}, время = {dt_cpsat:.1f}s")
        print(f"Метаэвристика: f = {f_meta}, время = {t_meta:.1f}s")
        print(f"Улучшение:    +{delta} ({100*delta/f_meta:.1f}%)")
        print(f"Общее время:  {t_meta + dt_cpsat:.1f}s")
    else:
        print("CP-SAT не нашёл решение")


if __name__ == '__main__':
    main()
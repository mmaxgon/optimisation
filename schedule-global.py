"""
файлы schedule_data.py, schedule-cpsat.py, schedule_workers.py, schedule-global.py. Файл python: C:\ProgramData\anaconda3\envs\opt

schedule-global.py — Глобальная эволюционная оптимизация расписания кинотеатра.

Алгоритм:
=========
1. Создаётся начальная популяция из N случайных допустимых решений.
2. Отбирается N_keep лучших решений (50% от N).
3. Итеративно:
   a. Мутация: в окрестности каждого лучшего решения генерируются новые
      кандидаты (search_neighborhood — зашумление параметров + SCIP + DP).
   b. Скрещивание: из пар лучших решений создаются потомки (uniform crossover).
   c. Иммиграция: каждые 10 итераций добавляется новое случайное решение.
   d. Отбор: объединяются лучшие + новые, сортируются, отбираются N_keep.
4. Остановка при стагнации (stagnation_limit итераций без улучшения)
   или при достижении max_iter.
5. (Опционально) Уточнение лучшего решения через полную модель CP-SAT
   с warm start (refine_with_cpsat).

Параллелизация:
  Все задачи (sample_initial, search_neighborhood) запускаются через
  ProcessPoolExecutor, используя все доступные ядра CPU.

Сравнение с прямым CP-SAT (schedule-cpsat.py):
  Метаэвристика: f ≈ 1200 за 20 секунд
  Прямой CP-SAT:  f ≈ 1237 за 600 секунд
  Разрыв: ~2.8% при 30x ускорении.
"""

from time import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import random
import numpy as np
from ortools.sat.python import cp_model
from schedule_data import *
from schedule_workers import sample_initial, search_neighborhood

############################################################################
# Параметры алгоритма
############################################################################

N = 20                    # Размер популяции (число параллельно развивающихся решений)
alpha = 0.95              # Коэффициент сжатия окрестности поиска:
                          #   шум = sigma * alpha^k, где k — номер итерации.
                          #   При alpha=0.95 после 100 итераций шум составляет
                          #   0.95^100 ≈ 0.006 от начального уровня.
max_iter = 1000           # Максимальное число итераций (если нет стагнации)
stagnation_limit = 100    # Число итераций без улучшения, после которого алгоритм останавливается
SEED = 42                 # Seed для воспроизводимости результатов

N_keep = N // 2           # Число лучших решений, сохраняемых на каждой итерации (элита)
sigma_y = 3.0             # Масштаб шума для предпочитаемых времён начала (в DP не используется,
                          #   но может применяться при crossover через search_neighborhood)


def refine_with_cpsat(schedule, time_limit=600):
    """
    Уточняет найденное метаэвристикой расписание с помощью полной модели CP-SAT.

    Строит монолитную модель оптимизации, аналогичную schedule-cpsat.py:
    - Булевы переменные x[зал, фильм, время] — начинается ли сеанс
    - Интервальные переменные для контроля непересечения
    - Целевая функция: sum(sales[фильм][время] * x[зал, фильм, время])

    Найденное метаэвристикой расписание используется как warm start (hint),
    что ускоряет сходимость CP-SAT к оптимуму.

    Args:
        schedule: расписание метаэвристики {(зал, i): {movie, start}}
        time_limit: лимит времени для CP-SAT (секунды)

    Returns:
        (solver, status, x, dt):
            solver — решатель CP-SAT после выполнения
            status — статус (OPTIMAL, FEASIBLE и др.)
            x — словарь булевых переменных модели
            dt — время решения в секундах
    """
    model = cp_model.CpModel()

    # --- Переменные уровня агрегации ---

    # movie_count_var[m] — общее число сеансов фильма m
    movie_count_var = {
        m: model.new_int_var(name=f"{m}_count", lb=movies[m].min, ub=movies[m].max)
        for m in movies
    }

    # hall_count_var[h] — общее число сеансов в зале h
    hall_count_var = {
        h: model.new_int_var(name=f"{h}_count", lb=0, ub=hall_max_shows[h])
        for h in halls
    }

    # movie_hall_count_var[h,m] — число сеансов фильма m в зале h
    movie_hall_count_var = {
        (h, m): model.new_int_var(name=f"{h}_{m}_count", lb=0, ub=hall_movie_max_shows[h, m])
        for m in movies for h in halls
    }

    # Баланс: сумма по залам = movie_count, сумма по фильмам = hall_count
    for m in movies:
        model.add(sum(movie_hall_count_var[h, m] for h in halls) == movie_count_var[m])
    for h in halls:
        model.add(sum(movie_hall_count_var[h, m] for m in movies) == hall_count_var[h])

    # --- Переменные уровня расписания ---

    # x[зал, фильм, время] = 1, если в зале h фильм m начинается в момент t.
    # Создаются только для совместимых пар (зал, фильм) и допустимых времён.
    x = {
        (h, m, t): model.new_bool_var(name=f"x[{h}, {m}, {t}]")
        for h in halls for m in hall_movies[h] for t in valid_starts[m]
    }

    # Интервал сеанса: [t, t + len + 1), где +1 — перерыв.
    # Optional interval: активен только если x[h,m,t] = 1.
    shows = {
        (h, m, t): model.new_optional_interval_var(
            start=t, size=movies[m].len + 1, end=t + movies[m].len + 1,
            is_present=x[h, m, t], name=f"show[{h}, {m}, {t}]"
        ) for (h, m, t) in x.keys()
    }

    # Связь: число активных сеансов фильма в зале = movie_hall_count
    for h in halls:
        for m in movies:
            if h in movie_halls[m]:
                model.add(sum(x[h, m, t] for t in valid_starts[m]) == movie_hall_count_var[h, m])
            else:
                model.add(movie_hall_count_var[h, m] == 0)

    # В каждый момент времени в зале начинается не больше одного сеанса
    for h in halls:
        for t in period:
            starts_at_t = [x[h, m, t] for m in hall_movies[h] if (h, m, t) in x]
            if starts_at_t:
                model.add(sum(starts_at_t) <= 1)

    # Сеансы в зале не пересекаются (через интервальные ограничения)
    for h in halls:
        hall_shows = [shows[h, m, t] for m in hall_movies[h] for t in valid_starts[m]]
        model.add_no_overlap(hall_shows)

    # --- Целевая функция: максимизация суммарных продаж ---
    model.maximize(sum(sales[m][t] * x[h, m, t] for (h, m, t) in x.keys()))

    # --- Warm start: подсказки из расписания метаэвристики ---

    # Множество сеансов из метаэвристики: {(зал, фильм, время_начала)}
    active_shows = set()
    for (h, i), info in schedule.items():
        active_shows.add((h, info["movie"], info["start"]))

    # Hint для булевых переменных: 1 если сеанс есть в расписании, иначе 0
    for (h, m, t) in x.keys():
        model.add_hint(x[h, m, t], 1 if (h, m, t) in active_shows else 0)

    # Hint для переменных уровня агрегации
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
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    start = time()
    status = solver.solve(model)
    dt = time() - start

    return solver, status, x, dt


def crossover_parents(params1, params2, alpha=0.5):
    """
    Uniform скрещивание двух родительских решений.

    Для каждой пары (зал, фильм) случайно выбирает значение от одного
    из родителей. В отличие от арифметического скрещивания (взвешенное
    среднее с округлением), сохраняет "хорошие" целочисленные значения
    родителей без усреднения.

    Args:
        params1: {(зал, фильм): кол-во} — первый родитель
        params2: {(зал, фильм): кол-во} — второй родитель
        alpha: не используется (оставлено для совместимости)

    Returns:
        dict: потомок — {(зал, фильм): кол-во}
    """
    child_params = {}
    for key in params1.keys():
        val1 = params1[key]
        val2 = params2[key] if key in params2 else 0
        # Uniform crossover: с вероятностью 50% берём от каждого родителя
        child_params[key] = val1 if random.random() < 0.5 else val2

    return child_params


def local_search_step(solution, max_perturb=2):
    """
    Выполняет локальный поиск вокруг заданного решения.

    Использует search_neighborhood с минимальным шумом (k=0, alpha=1.0),
    что означает поиск в очень малой окрестности текущего решения.

    Args:
        solution: (params, schedule_dict, score) — текущее решение
        max_perturb: не используется (зарезервировано для будущих улучшений)

    Returns:
        tuple: новое решение после локального поиска
    """
    schedule_data, schedule_dict, score = solution
    return search_neighborhood(solution, k=0, alpha=1.0, sigma_y=0.5)


def main():
    """
    Основная функция запуска глобальной оптимизации расписания.

    Последовательность работы:
    1. Инициализация популяции (N случайных решений через sample_initial)
    2. Отбор N_keep лучших (элита)
    3. Основной цикл:
       - Мутация: search_neighborhood для каждого элитного решения
       - Скрещивание: uniform crossover + search_neighborhood для потомков
       - Иммиграция: sample_initial каждые 10 итераций
       - Отбор: N_keep лучших из объединённого множества
    4. Вывод результатов
    """
    # Фиксируем seed для воспроизводимости
    np.random.seed(SEED)
    random.seed(SEED)

    # Число параллельных процессов = min(число_ядер_CPU, N)
    n_workers = min(os.cpu_count() or 1, N)

    print("=" * 60)
    print(f"Алгоритм: N={N}, alpha={alpha}, max_iter={max_iter}, workers={n_workers}")
    print("=" * 60)

    t_total = time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # ================================================================
        # Фаза 1: Начальная выборка (параллельная генерация популяции)
        # ================================================================
        population = []
        max_attempts = N * 5   # Максимальное число попыток (не все random решения допустимы)
        batch_size = n_workers * 3  # Размер батча для параллельной подачи
        submitted = 0
        while len(population) < N and submitted < max_attempts:
            n_submit = min(batch_size, max_attempts - submitted)
            # Запускаем n_submit задач параллельно
            futures = [executor.submit(sample_initial) for _ in range(n_submit)]
            submitted += n_submit
            # Собираем результаты по мере завершения
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception:
                    continue
                if result is not None:
                    population.append(result)
                if len(population) >= N:
                    break

        # Сортируем по убыванию целевой функции (максимизация продаж)
        population.sort(key=lambda r: r[2], reverse=True)
        # Элита: N_keep лучших решений
        n_keep = min(N_keep, len(population))
        best = list(population[:n_keep])
        # История всех вычисленных решений (для статистики)
        history = list(population)

        print(f"Начальная выборка: {len(population)} допустимых точек")
        print(f"Лучшее f = {best[0][2]}")

        # ================================================================
        # Фаза 2: Итеративное улучшение
        # ================================================================
        prev_best_f = best[0][2]    # Лучшее значение на предыдущей итерации
        stagnation_count = 0         # Счётчик итераций без улучшения

        for k in range(1, max_iter + 1):
            futures = []

            # --- Оператор 1: Мутация (поиск в окрестности лучших) ---
            # Для каждого из N_keep лучших решений генерируем нового кандидата
            # через зашумление параметров + SCIP + мульти-кандидатный DP
            for j in range(n_keep):
                futures.append(
                    executor.submit(search_neighborhood, best[j], k, alpha, sigma_y)
                )

            # --- Оператор 2: Скрещивание (crossover) ---
            # Создаём потомков от пар лучших родителей.
            # Берём параметры movie_hall_count от двух родителей,
            # применяем uniform crossover, затем решаем через search_neighborhood.
            n_crossover = N_keep // 3  # ~30% от размера элиты
            if len(best) >= 2:
                for _ in range(n_crossover):
                    # Случайный выбор двух различных родителей из элиты
                    p1, p2 = random.sample(best, 2)
                    try:
                        # Uniform crossover параметров (зал, фильм) → число сеансов
                        child_params = crossover_parents(p1[0], p2[0], alpha=0.5)
                        # Фиктивное решение: параметры есть, расписания нет
                        dummy_solution = (child_params, None, 0.0)
                        # Решаем уровни 2+3 для потомка (с уменьшенным шумом sigma_y*0.7)
                        futures.append(
                            executor.submit(search_neighborhood, dummy_solution, k, alpha, sigma_y * 0.7)
                        )
                    except Exception:
                        continue

            # --- Оператор 3: Иммиграция ---
            # Каждые 10 итераций добавляем одно полностью случайное решение
            # для поддержания разнообразия популяции
            if k % 10 == 0:
                futures.append(executor.submit(sample_initial))

            # --- Сбор результатов ---
            new_points = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        new_points.append(result)
                except Exception:
                    continue

            # --- Отбор: объединяем элиту с новыми кандидатами ---
            combined = best + new_points
            combined.sort(key=lambda r: r[2], reverse=True)  # Сортировка по продажам
            best = combined[:n_keep]  # Оставляем N_keep лучших
            history.extend(new_points)

            # --- Проверка улучшения ---
            current_best_f = best[0][2]
            if current_best_f > prev_best_f:
                stagnation_count = 0     # Улучшение найдено — сбрасываем стагнацию
            else:
                stagnation_count += 1    # Нет улучшения — увеличиваем счётчик
            prev_best_f = current_best_f

            # Выводим статистику итерации
            print(f"Итерация {k:3d}: f = {current_best_f:6.1f} | "
                  f"новых = {len(new_points):2d} | стагнация = {stagnation_count} | "
                  f"t = {time() - t_total:.1f}s")

            # Проверка условия остановки по стагнации
            if stagnation_count >= stagnation_limit:
                print(f"Остановка: стагнация {stagnation_limit} итераций")
                break

    # ================================================================
    # Результаты
    # ================================================================
    f_meta = best[0][2]             # Лучшее значение целевой функции
    t_meta = time() - t_total       # Общее время работы
    print("\n" + "=" * 60)
    print(f"МЕТАЭВРИСТИКА: f = {f_meta}, время = {t_meta:.1f}s")
    print(f"Всего вычислено точек: {len(history)}")

    # --- (Опционально) Уточнение через CP-SAT с warm start ---
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

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
alpha = 0.9               # коэффициент сжатия окрестности поиска
max_iter = 30             # максимальное число итераций
stagnation_limit = 10      # останов при стагнации (итераций без улучшения)
SEED = 42

N_keep = N // 2  # число лучших точек, сохраняемых на каждой итерации
sigma_y = 3.0  # масштаб шума для предпочтительных времён начала (итерации)


#############################################################
# Warm start: уточнение через полную модель CP-SAT
#############################################################

def refine_with_cpsat(schedule, time_limit=600):
    """
    Полная модель CP-SAT (как в schedule-cpsat.py) с hint solution
    из расписания, найденного метаэвристикой.
    """
    model = cp_model.CpModel()

    # --- Переменные ---
    movie_count_var = {
        m: model.new_int_var(name=f"{m}_count", lb=movies[m].min, ub=movies[m].max)
        for m in movies
    }
    hall_count_var = {
        h: model.new_int_var(name=f"{h}_count", lb=0, ub=hall_max_shows[h])
        for h in halls
    }
    movie_hall_count_var = {
        (h, m): model.new_int_var(name=f"{h}_{m}_count", lb=0, ub=hall_movie_max_shows[h, m])
        for m in movies for h in halls
    }

    for m in movies:
        model.add(sum(movie_hall_count_var[h, m] for h in halls) == movie_count_var[m])
    for h in halls:
        model.add(sum(movie_hall_count_var[h, m] for m in movies) == hall_count_var[h])

    x = {
        (h, m, t): model.new_bool_var(name=f"x[{h}, {m}, {t}]")
        for h in halls for m in hall_movies[h] for t in valid_starts[m]
    }
    shows = {
        (h, m, t): model.new_optional_interval_var(
            start=t, size=movies[m].len + 1, end=t + movies[m].len + 1,
            is_present=x[h, m, t], name=f"show[{h}, {m}, {t}]"
        ) for (h, m, t) in x.keys()
    }

    for h in halls:
        for m in movies:
            if h in movie_halls[m]:
                model.add(sum(x[h, m, t] for t in valid_starts[m]) == movie_hall_count_var[h, m])
            else:
                model.add(movie_hall_count_var[h, m] == 0)

    for h in halls:
        for t in period:
            starts_at_t = [x[h, m, t] for m in hall_movies[h] if (h, m, t) in x]
            if starts_at_t:
                model.add(sum(starts_at_t) <= 1)

    for h in halls:
        hall_shows = [shows[h, m, t] for m in hall_movies[h] for t in valid_starts[m]]
        model.add_no_overlap(hall_shows)

    model.maximize(sum(sales[m][t] * x[h, m, t] for (h, m, t) in x.keys()))

    # --- warm start: hint из расписания метаэвристики ---
    active_shows = set()
    for (h, i), info in schedule.items():
        active_shows.add((h, info["movie"], info["start"]))

    for (h, m, t) in x.keys():
        model.add_hint(x[h, m, t], 1 if (h, m, t) in active_shows else 0)

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


def main():
    np.random.seed(SEED)
    random.seed(SEED)

    n_workers = min(os.cpu_count() or 1, N)

    # --- Шаги 2-6: начальное исследование (параллельно) ---
    print("=" * 60)
    print(f"Алгоритм: N={N}, alpha={alpha}, max_iter={max_iter}, workers={n_workers}")
    print("=" * 60)

    t_total = time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Начальная выборка: батчи по n_workers*3 задач
        population = []
        max_attempts = N * 5
        batch_size = n_workers * 3
        submitted = 0
        while len(population) < N and submitted < max_attempts:
            n_submit = min(batch_size, max_attempts - submitted)
            futures = [executor.submit(sample_initial) for _ in range(n_submit)]
            submitted += n_submit
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception:
                    continue
                if result is not None:
                    population.append(result)
                if len(population) >= N:
                    break

        population.sort(key=lambda r: r[2], reverse=True)  # maximize
        n_keep = min(N_keep, len(population))
        best = list(population[:n_keep])
        history = list(population)

        print(f"Начальная выборка: {len(population)} допустимых точек")
        print(f"Лучшее f = {best[0][2]}")

        # --- Шаги 7-11: итерации (параллельный поиск в окрестности) ---
        prev_best_f = best[0][2]
        stagnation_count = 0

        for k in range(1, max_iter + 1):
            futures = [executor.submit(search_neighborhood, best[j], k, alpha, sigma_y)
                       for j in range(n_keep)]
            new_points = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception:
                    continue
                if result is not None:
                    new_points.append(result)

            # Шаг 6: объединяем, отбираем n_keep лучших
            combined = best + new_points
            combined.sort(key=lambda r: r[2], reverse=True)
            best = combined[:n_keep]
            history.extend(new_points)

            # Проверка улучшения
            current_best_f = best[0][2]
            if current_best_f > prev_best_f:
                stagnation_count = 0
            else:
                stagnation_count += 1
            prev_best_f = current_best_f

            print(f"Итерация {k:3d}: f = {current_best_f:6.1f} | "
                  f"новых = {len(new_points):2d} | стагнация = {stagnation_count} | "
                  f"t = {time() - t_total:.1f}s")

            if stagnation_count >= stagnation_limit:
                print(f"Остановка: стагнация {stagnation_limit} итераций")
                break

    # --- Результат метаэвристики ---
    f_meta = best[0][2]
    t_meta = time() - t_total
    print("\n" + "=" * 60)
    print(f"МЕТАЭВРИСТИКА: f = {f_meta}, время = {t_meta:.1f}s")
    print(f"Всего вычислено точек: {len(history)}")

    # --- Уточнение через CP-SAT с тёплым стартом ---
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


"""
  ┌─────────────────┬────────────────────┬────────────────────────────┬──────────┐
  │                 │   Global (N=20)    │            SCIP            │  CP-SAT  │
  ├─────────────────┼────────────────────┼────────────────────────────┼──────────┤
  │ f (цель)        │ 862                │ 84                         │ 1235     │
  ├─────────────────┼────────────────────┼────────────────────────────┼──────────┤
  │ Время           │ 5.1s               │ 599s                       │ 614s     │
  ├─────────────────┼────────────────────┼────────────────────────────┼──────────┤
  │ Статус          │ останов: стагнация │ timelimit                  │ FEASIBLE │
  ├─────────────────┼────────────────────┼────────────────────────────┼──────────┤
  │ Залы с сеансами │ все 10             │ только 2 (hall_9, hall_10) │ все 10   │
  └─────────────────┴────────────────────┴────────────────────────────┴──────────┘

SCIP практически не справился — Big-M формулировка для no-overlap слишком слаба для релаксации,
SCIP за 10 минут нашёл только тривиальное расписание в 2 залах.

CP-SAT нашёл лучшее решение (1235) за 10 минут, но статус FEASIBLE — оптимум не достигнут.

Global получил 70% качества CP-SAT (862 vs 1235) за ~0.8% времени (5.1s vs 614s).
Это нормальное соотношение для метаэвристики — она не оптимальна, но на порядки быстрее.
"""

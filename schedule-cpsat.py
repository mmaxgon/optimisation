"""
schedule-cpsat.py — Прямая модель CP-SAT для оптимизации расписания кинотеатра.

Монолитная модель: все переменные (зал × фильм × время) создаются одновременно,
ограничения непересечения задаются через интервальные переменные, целевая функция
максимизирует суммарные продажи напрямую.

Используется как baseline для сравнения с метаэвристикой (schedule-global.py).

Результат: f ≈ 1237 за ~600 секунд (OPTIMAL или FEASIBLE).
Метаэвристика: f ≈ 1202 за ~20 секунд (разрыв ~2.8%).
"""

from time import time
from ortools.sat.python import cp_model
from schedule_data import *

############################################################################
# Модель
############################################################################

model = cp_model.CpModel()

############################################################################
# Переменные + Ограничения
############################################################################

# --- Переменные уровня агрегации ---

# movie_count[m] — общее число сеансов фильма m по всем залам
movie_count = {
	m: model.new_int_var(
		name=f"{m}_count",
		lb=movies[m].min,   # Фильм должен быть показан хотя бы min раз
		ub=movies[m].max    # Не более max раз
	) for m in movies
}

# hall_count[h] — общее число сеансов в зале h
hall_count = {
	h: model.new_int_var(
		name=f"{h}_count",
		lb=0,
		ub=hall_max_shows[h]
	) for h in halls
}

# movie_hall_count[h,m] — число сеансов фильма m в зале h
# Для несовместимых пар верхняя граница = 0 (сеансы невозможны)
movie_hall_count = {
	(h, m): model.new_int_var(
		name=f"{h}_{m}_count",
		lb=0,
		ub=hall_movie_max_shows[h, m]
	) for m in movies for h in halls
}

# Баланс: сумма movie_hall_count по залам = movie_count
for m in movies:
	cons = model.add(sum(movie_hall_count[h, m] for h in halls) == movie_count[m])

# Баланс: сумма movie_hall_count по фильмам = hall_count
for h in halls:
	cons = model.add(sum(movie_hall_count[h, m] for m in movies) == hall_count[h])

# --- Переменные уровня расписания ---

# x[h,m,t] = 1, если в зале h фильм m начинается в момент времени t.
# Создаются только для совместимых (зал, фильм) пар и допустимых времён начала
# (сеанс должен закончиться не позднее T).
x = {
	(h, m, t): model.new_bool_var(name=f"x[{h}, {m}, {t}]")
	for h in halls for m in hall_movies[h] for t in valid_starts[m]
}

# Интервал сеанса: размер = len + 1 (длительность + перерыв 5 минут).
# Optional interval — активен только если x[h,m,t] = 1.
# Используется для ограничения непересечения через add_no_overlap.
shows = {
	(h, m, t): model.new_optional_interval_var(
		start=t,
		size=movies[m].len + 1,      # Длительность фильма + перерыв
		end=t + movies[m].len + 1,
		is_present=x[h, m, t],        # Интервал существует только если x=1
		name=f"show[{h}, {m}, {t}]"
	) for (h, m, t) in x.keys()
}

# Согласование: число активных сеансов = movie_hall_count
for h in halls:
	for m in movies:
		if h in movie_halls[m]:
			cons = model.add(sum(x[h, m, t] for t in valid_starts[m]) == movie_hall_count[h, m])
		else:
			cons = model.add(movie_hall_count[h, m] == 0)

# В каждый момент времени в зале начинается не больше одного сеанса
# (дополнительное ограничение, усиливает no_overlap)
for h in halls:
	for t in period:
		starts_at_t = [x[h, m, t] for m in hall_movies[h] if (h, m, t) in x]
		if starts_at_t:
			cons = model.add(sum(starts_at_t) <= 1)

# Сеансы не пересекаются (основное ограничение через интервальные переменные)
for h in halls:
	hall_shows = [shows[h, m, t] for m in hall_movies[h] for t in valid_starts[m]]
	cons = model.add_no_overlap(hall_shows)

############################################################################
# Целевая функция
############################################################################

# Максимизируем суммарные ожидаемые продажи:
# sum(sales[фильм][время_начала] * x[зал, фильм, время_начала])
model.maximize(sum(sales[m][t] * x[h, m, t] for (h, m, t) in x.keys()))

############################################################################
# Решение
############################################################################

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 600  # Лимит: 10 минут

start_time = time()
status = solver.solve(model)
end_time = time()

############################################################################
# Вывод результата
############################################################################

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
	# Печатаем все назначенные сеансы
	for (h, m, t) in x.keys():
		if solver.boolean_value(x[h, m, t]):
			print(f"hall {h}, time {t}, movie {m}")
else:
	print('No solution found.')

# Статус решения (OPTIMAL / FEASIBLE / INFEASIBLE и др.)
print(solver.status_name(status))
# Значение целевой функции (суммарные продажи)
print(solver.objective_value)
# Время решения
print(end_time - start_time)

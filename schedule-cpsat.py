"""
https://developers.google.com/optimization/reference/python/sat/python/cp_model
https://github.com/google/or-tools/blob/stable/ortools/sat/samples/
"""
from time import time
from ortools.sat.python import cp_model
from schedule_data import *

############################################################################
# Модель
############################################################################

model = cp_model.CpModel()

############################################################################
# Decision vars + Constraints
############################################################################

# Число сеансов данного фильма во всех залах
movie_count = {
	m: model.new_int_var(
		name=f"{m}_count",
		lb=movies[m].min,
		ub=movies[m].max
	) for m in movies
}

# Число сеансов всех фильмов в данном зале
hall_count = {
	h: model.new_int_var(
		name=f"{h}_count",
		lb=0,
		ub=hall_max_shows[h]
	) for h in halls
}

# Число сеансов данного фильма в данном зале
movie_hall_count = {
	(h, m): model.new_int_var(
		name=f"{h}_{m}_count",
		lb=0,
		ub=hall_movie_max_shows[h, m] # Фильмы только в поддерживаемых его формат залах
	) for m in movies for h in halls
}

# Согласовываем количество сеансов фильмов в залах
for m in movies:
	cons = model.add(sum(movie_hall_count[h, m] for h in halls) == movie_count[m])
for h in halls:
	cons = model.add(sum(movie_hall_count[h, m] for m in movies) == hall_count[h])

# Начинается ли сеанс данного фильма в данном зале в определенный момент времени
# Только для совместимых (зал, фильм) и допустимых времён начала (сеанс помещается в T)
x = {
	(h, m, t): model.new_bool_var(name=f"x[{h}, {m}, {t}]")
	for h in halls for m in hall_movies[h] for t in valid_starts[m]
}
# Интервал сеанса (размер = len + 1 шаг gap)
shows = {
	(h, m, t): model.new_optional_interval_var(
		start=t,
		size=movies[m].len + 1,
		end=t + movies[m].len + 1,
		is_present=x[h, m, t],
		name=f"show[{h}, {m}, {t}]"
	) for (h, m, t) in x.keys()
}

# Согласовываем назначение сеансов с числом сеансов каждого фильма в каждом зале
for h in halls:
	for m in movies:
		if h in movie_halls[m]:
			cons = model.add(sum(x[h, m, t] for t in valid_starts[m]) == movie_hall_count[h, m])
		else:
			cons = model.add(movie_hall_count[h, m] == 0)

# в каждом зале в каждый момент времени начинается не больше одного сеанса
for h in halls:
	for t in period:
		starts_at_t = [x[h, m, t] for m in hall_movies[h] if (h, m, t) in x]
		if starts_at_t:
			cons = model.add(sum(starts_at_t) <= 1)

# сеансы не пересекаются (ограничение через интервалы)
for h in halls:
	hall_shows = [shows[h, m, t] for m in hall_movies[h] for t in valid_starts[m]]
	cons = model.add_no_overlap(hall_shows)

############################################################################
# Objective
############################################################################

model.maximize(sum(sales[m][t] * x[h, m, t] for (h, m, t) in x.keys()))

############################################################################
# Solve
############################################################################
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 600
start_time = time()
status = solver.solve(model)
end_time = time()

############################################################################
# Solution
############################################################################
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
	for (h, m, t) in x.keys():
		if solver.boolean_value(x[h, m, t]):
			print(f"hall {h}, time {t}, movie {m}")
else:
	print('No solution found.')

print(solver.status_name(status))
print(solver.objective_value)
print(end_time - start_time)

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
	model.add(sum(movie_hall_count[h, m] for h in halls) == movie_count[m])
for h in halls:
	model.add(sum(movie_hall_count[h, m] for m in movies) == hall_count[h])

# Фильмы только в поддерживаемых его формат залах
for m in movies:
	allowed_h = movie_halls[m]
	for h in halls:
		if not (h in allowed_h):
			model.add(movie_hall_count[h, m] == 0)

# Начинается ли сеанс данного фильма в данном зале в определенный момент времени
x = {(h, m, t): model.new_bool_var(name=f"x[{h}, {m}, {t}]") for h in halls for m in movies for t in period}
# Интервал сеанса
shows = {(h, m, t): model.new_optional_interval_var(
	start=t,
	size=movies[m].len + 1,
	end=t + movies[m].len + 1,
	is_present=x[h, m, t],
	name=f"show[{h}, {m}, {t}]"
) for h in halls for m in movies for t in period}

# Согласовываем назначение сеансов с числом сеансов каждого фильма в каждом зале
for h in halls:
	for m in movies:
		model.add(sum(x[h, m, t] for t in period) == movie_hall_count[h, m])

# в каждом зале в каждый момент времени начинается не больше одного сеанса
for h in halls:
 	for t in period:
		 model.add(sum(x[h, m, t] for m in movies) <= 1)

# сеансы не пересекаются (ограничение через интервалы)
for h in halls:
 	model.add_no_overlap(shows[h, m, t] for m in movies for t in period)

# сеансы не пересекаются (ограничение через BIG_M)
# for h in halls:
# 	for m in movies:
# 		d = movies[m].len
# 		for tstart in range(T):
# 			tend = tstart + np.min([T - tstart - 1, d])
# 			#print(d, tstart, tend)
# 			model.add(d * x[h, m, tstart] + sum(x[h, m1, t] for t in range(tstart + 1, tend + 1) for m1 in movies) <= d)

############################################################################
# Objective
############################################################################

model.maximize(sum(sales[m][t] * x[h, m, t] for h in halls for m in movies for t in period))

############################################################################
# Solve
############################################################################
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60.0
start_time = time()
status = solver.solve(model)
end_time = time()

############################################################################
# Solution
############################################################################
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
	for ix_h, h in enumerate(halls):
		for ix_t, t in enumerate(period):
			for ix_m, m in enumerate(movies):
				if solver.boolean_value(x[h, m, t]):
					print(f"hall {h}, time {t}, movie {m}")
else:
	print('No solution found.')

print(solver.status_name(status))
print(solver.objective_value)
print(end_time - start_time)

#############################################################################
def show_var(var_name):
	print({ix: solver.value(var_name[ix]) for ix in var_name})

# show_var(movie_count)
# show_var(hall_count)
# show_var(movie_hall_count)

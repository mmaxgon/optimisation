from time import time
import numpy as np
import pyscipopt as scip
from schedule_data import *

############################################################################
# Модель
############################################################################

model = scip.Model("SCIP Schedule")

############################################################################
# Decision vars + Constraints
############################################################################

# Число сеансов данного фильма во всех залах
movie_count = {
	m: model.addVar(
		name=f"{m}_count",
		lb=movies[m].min,
		ub=movies[m].max,
		vtype="I"
	) for m in movies
}

# Число сеансов всех фильмов в данном зале
hall_count = {
	h: model.addVar(
		name=f"{h}_count",
		lb=0,
		ub=hall_max_shows[h],
		vtype="I"
	) for h in halls
}

# Число сеансов данного фильма в данном зале
movie_hall_count = {
	(h, m): model.addVar(
		name=f"{h}_{m}_count",
		lb=0,
		ub=hall_movie_max_shows[h, m],
		vtype="I"
	) for m in movies for h in halls
}

# Согласовываем количество сеансов фильмов в залах
for m in movies:
	cons = model.addCons(sum(movie_hall_count[h, m] for h in halls) == movie_count[m])
for h in halls:
	cons = model.addCons(sum(movie_hall_count[h, m] for m in movies) == hall_count[h])

# Начинается ли сеанс данного фильма в данном зале в определенный момент времени
# Только для совместимых (зал, фильм) и допустимых времён начала (сеанс помещается в T)
x = {
	(h, m, t): model.addVar(name=f"x[{h}, {m}, {t}]", vtype="B")
	for h in halls for m in hall_movies[h] for t in valid_starts[m]
}

# Согласовываем назначение сеансов с числом сеансов каждого фильма в каждом зале
for h in halls:
	for m in movies:
		if h in movie_halls[m]:
			cons = model.addCons(sum(x[h, m, t] for t in valid_starts[m]) == movie_hall_count[h, m])
		else:
			cons = model.addCons(movie_hall_count[h, m] == 0)

# в каждом зале в каждый момент времени начинается не больше одного сеанса
for h in halls:
	for t in period:
		starts_at_t = [x[h, m, t] for m in hall_movies[h] if (h, m, t) in x]
		if starts_at_t:
			cons = model.addCons(sum(starts_at_t) <= 1)

# сеансы не пересекаются (ограничение через BIG_M)
# gap = 1 шаг (5 мин) между сеансами уже учтён: блокируем [tstart+1, tstart+d]
# for h in halls:
# 	for m in hall_movies[h]:
# 		d = movies[m].len
# 		for tstart in valid_starts[m]:
# 			block = [
# 				x[h, m1, t]
# 				for m1 in hall_movies[h]
# 				for t in range(tstart + 1, tstart + d + 1)
# 				if (h, m1, t) in x
# 			]
# 			cons = model.addCons(d * x[h, m, tstart] + sum(block) <= d)
for h in halls:
	for m in hall_movies[h]:
		d = movies[m].len
		for tstart in valid_starts[m]:
		# если сеанс начался в tstart, то блокируем [tstart+1, tstart+d]
			for t in range(tstart + 1, min(tstart + d + 1, period[-1] - 1)):
				block = [x[h, m1, t] for m1 in hall_movies[h] if (h, m1, t) in x]
				if block:
					cons = model.addCons(1 - x[h, m, tstart] >= sum(block))

############################################################################
# Objective
############################################################################

model.setObjective(sum(sales[m][t] * x[h, m, t] for (h, m, t) in x.keys()), sense='maximize')

############################################################################
# Solve
############################################################################
model.setParam("display/verblevel", 1)
model.setParam('limits/time', 600)

start_time = time()
model.optimize()
sol = model.getBestSol()
end_time = time()

status = model.getStatus()
print(status)

############################################################################
# Solution
############################################################################
if status in ("optimal", "gaplimit", "timelimit"):
	for (h, m, t) in x.keys():
		if sol[x[h, m, t]] > 0:
			print(f"hall: {h}, start: {t}, movie: {m}, duration: {movies[m].len}")
	print(model.getSolObjVal(sol))
else:
	print('No solution found.')

print(end_time - start_time)

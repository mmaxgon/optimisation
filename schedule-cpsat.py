"""
https://developers.google.com/optimization/reference/python/sat/python/cp_model
https://github.com/google/or-tools/blob/stable/ortools/sat/samples/
"""
import numpy as np
from ortools.sat.python import cp_model

############################################################################
# Data
############################################################################

T = 24
period = range(T)

movie_formats = set(["A", "B"])

movies = {
	"movie_1" : {"len" : 5, "min": 3, "max": 5, "format": "A"},	
	"movie_2" : {"len" : 3, "min": 4, "max": 10, "format": "A"},	
	"movie_3" : {"len" : 7, "min": 2, "max": 4, "format": "B"}
}
movie_names = list(movies.keys())

halls = {
	"hall_1": {"supported_formats": ["A"], "max_shows": 15},
	"hall_2": {"supported_formats": ["A"], "max_shows": 12},
	"hall_3": {"supported_formats": ["A", "B"], "max_shows": 10},
}
hall_names = list(halls.keys())

movie_halls = {movie : [hall for hall in halls if movies[movie]["format"] in halls[hall]["supported_formats"]] for movie in movies}
hall_movies = {hall : [movie for movie in movies if movies[movie]["format"] in halls[hall]["supported_formats"]] for hall in halls}

sales = {
	"movie_1": [1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1,1,1,1,1,1,1,1],
	"movie_2": [1,1,1,1,1,2,3,4,5,6,7,8,9,8,7,6,6,5,4,3,3,1,1,1],
	"movie_3": [9,9,9,8,7,6,5,4,3,2,1,1,1,1,1,1,2,3,4,5,6,7,6,5],
}

############################################################################
# Модель
############################################################################

model = cp_model.CpModel()

############################################################################
# Decision vars
############################################################################
x = [[[model.NewBoolVar("x[{}, {}, {}]".format(h, m, t)) for t in period] for m in movies] for h in halls]
shows = [[[model.NewOptionalIntervalVar(t, movies[m]["len"] + 1, t + movies[m]["len"] + 1, x[ix_h][ix_m][t], "show[{}, {}, {}]".format(h, m, t)) for t in period] for ix_m, m in enumerate(movies)] for ix_h, h in enumerate(halls)]
############################################################################
# Constraints
############################################################################
NBIG = 100

# Максиммальное и минимальное число фильмов
for ix_m, movie in enumerate(movies):
	model.Add(sum(x[ix_h][ix_m][t] for ix_h in range(len(halls)) for t in period) <= movies[movie]["max"])
	model.Add(sum(x[ix_h][ix_m][t] for ix_h in range(len(halls)) for t in period) >= movies[movie]["min"])

# Фильмы только в поддерживаемых формат заллах
for ix_m, m in enumerate(movies):
	allowed_h = movie_halls[m]
	for ix_h, h in enumerate(halls):
		if not (h in allowed_h):
			model.Add(sum(x[ix_h][ix_m][t] for t in period) == 0)
			
"""
Непересечение сеансов можно задать через линейные логические неравенства с использованием NBIG,
а можно через интервалы и ограничение на них NoOverLap
"""		
# в каждом зале в каждый момент времени начинается не больше одного сеанса
for ix_h, h in enumerate(halls):
 	for t in period:
		 model.Add(sum(x[ix_h][ix_m][t] for ix_m in range(len(movies))) <= 1)
# сеансы не пересекаются
for ix_h, h in enumerate(halls):
	for ix_m, m in enumerate(movies):
		d = movies[m]["len"]
		for tstart in range(T):
			tend = tstart + np.min([T - tstart - 1, d])
			#print(d, tstart, tend)
			model.Add(NBIG * x[ix_h][ix_m][tstart] + sum(x[ix_h][ix_m1][t] for t in range(tstart, tend + 1) for ix_m1, m1 in enumerate(movies) if m1 != m) <= NBIG)
			model.Add(NBIG * x[ix_h][ix_m][tstart] + sum(x[ix_h][ix_m][t] for t in range(tstart + 1, tend + 1)) <= NBIG)

# # сеансы не пересекаются (ограничение через интервалы)
# for ix_h, h in enumerate(halls):
#  	model.AddNoOverlap(shows[ix_h][ix_m][t] for ix_m, m in enumerate(movies) for t in period)
############################################################################
# Objective
############################################################################
model.Maximize(sum(sales[m][t] * x[ix_h][ix_m][t] for ix_h, h in enumerate(halls) for ix_m, m in enumerate(movies) for t in period))

############################################################################
# Solve
############################################################################
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60.0
status = solver.Solve(model)

############################################################################
# Solution
############################################################################
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
	for ix_h, h in enumerate(halls):
		for ix_t, t in enumerate(period):
			for ix_m, m in enumerate(movies):
				if solver.BooleanValue(x[ix_h][ix_m][ix_t]):
					print("hall {}, time {}, movie {}".format(h, t, m))
else:
	print('No solution found.')

print(solver.ObjectiveValue())
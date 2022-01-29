from time import time
import numpy as np
from docplex.cp.model import CpoModel

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

model = CpoModel()

############################################################################
# Decision vars
############################################################################
shows = {(h, m, t):
	         model.interval_var(
		         start=t,
		         end=t + movies[m]["len"] + 1,
		         size=movies[m]["len"] + 1,
		         optional=True, name="show[{}, {}, {}]".format(h, m, t)
	         ) for t in period for ix_m, m in enumerate(movies) for ix_h, h in enumerate(halls)}
############################################################################
# Constraints
############################################################################
# Максиммальное и минимальное число фильмов
for ix_m, movie in enumerate(movies):
	model.add(sum(model.presence_of(shows[(h, movie, t)]) for h in halls for t in period) <= movies[movie]["max"])
	model.add(sum(model.presence_of(shows[(h, movie, t)]) for h in halls for t in period) >= movies[movie]["min"])

# Фильмы только в поддерживаемых формат заллах
for ix_m, m in enumerate(movies):
	allowed_h = movie_halls[m]
	for ix_h, h in enumerate(halls):
		if not (h in allowed_h):
			model.add(sum(model.presence_of(shows[(h, m, t)]) for t in period) == 0)

# в каждом зале в каждый момент времени начинается не больше одного сеанса
for ix_h, h in enumerate(halls):
 	for t in period:
		 model.add(sum(model.presence_of(shows[(h, m, t)]) for m in movies) <= 1)

# # сеансы не пересекаются (ограничение через интервалы)
for ix_h, h in enumerate(halls):
 	model.add(model.no_overlap([shows[(h, m, t)] for ix_m, m in enumerate(movies) for t in period]))
############################################################################
# Objective
############################################################################
model.maximize(
	sum(
		sales[m][t] * (model.presence_of(shows[(h, m, t)]))
		for ix_h, h in enumerate(halls)
		for ix_m, m in enumerate(movies)
		for t in period
	)
)

############################################################################
# Solve
############################################################################
start_time = time()
msol = model.solve(TimeLimit=10, trace_log=False, Workers="Auto")
end_time = time()

############################################################################
# Solution
############################################################################
if msol.get_solve_status() == "Optimal" or msol.get_solve_status() == "Feasible":
	for ix_h, h in enumerate(halls):
		for ix_t, t in enumerate(period):
			for ix_m, m in enumerate(movies):
				if len(msol[shows[h, m, t]]) > 0:
					print("hall {}, time {}, movie {}".format(h, t, m))
else:
	print('No solution found.')

print(msol.get_solve_status())
print(msol.get_objective_value())
print(end_time - start_time)
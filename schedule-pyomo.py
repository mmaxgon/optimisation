from time import time
import numpy as np
from dataclasses import dataclass
import pyomo.environ as py

############################################################################
# Data
############################################################################

T = 24
period = range(T)

# Форматы фильмов
movie_formats = set(["A", "B"])

@dataclass
class Movie:
	len: int            # длительность
	min: int            # минимальное число в расписании
	max: int            # максимальное число в расписании
	format: str         # формат
	def __post_init__(self):
		assert(self.format in movie_formats)

movies = {
	"movie_1": Movie(len=5, min=3, max=5, format="A"),
	"movie_2": Movie(len=3, min=4, max=10, format="A"),
	"movie_3": Movie(len=7, min=2, max=4, format="B"),
}
movie_names = list(movies.keys())

@dataclass
class Hall:
	supported_formats: list[movie_formats]      # поддерживаемые форматы
	max_shows: int                              # максимальное число сеансов
	def __post_init__(self):
		assert(type(self.supported_formats) is list)
		assert(all([f in movie_formats for f in self.supported_formats]))

halls = {
	"hall_1": Hall(supported_formats=["A"], max_shows= 15),
	"hall_2": Hall(supported_formats=["A"], max_shows=12),
	"hall_3": Hall(supported_formats=["A", "B"], max_shows=10),
}
hall_names = list(halls.keys())

# допустимые комбинации залов и фильмов
movie_halls = {movie : [hall for hall in halls if movies[movie].format in halls[hall].supported_formats] for movie in movies}
hall_movies = {hall : [movie for movie in movies if movies[movie].format in halls[hall].supported_formats] for hall in halls}

# спрос на фильм в зависимости от времени начала показа
sales = {
	"movie_1": [1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1,1,1,1,1,1,1,1],
	"movie_2": [1,1,1,1,1,2,3,4,5,6,7,8,9,8,7,6,6,5,4,3,3,1,1,1],
	"movie_3": [9,9,9,8,7,6,5,4,3,2,1,1,1,1,1,1,2,3,4,5,6,7,6,5],
}

############################################################################
# Модель
############################################################################

model = py.ConcreteModel()

############################################################################
# Decision vars
############################################################################

model.x = py.Var(hall_names, movie_names, period, domain=py.Binary)
# model.x = py.Var(hall_names, movie_names, period, domain=py.Reals, bounds=(0, 1))

############################################################################
# Constraints
############################################################################
NBIG = 100

# Максиммальное и минимальное число фильмов
model.show_ranges = py.ConstraintList()
for m in movie_names:
	model.show_ranges.add(sum(model.x[h, m, t] for h in hall_names for t in period) <= movies[m].max)
	model.show_ranges.add(sum(model.x[h, m, t] for h in hall_names for t in period) >= movies[m].min)

# Фильмы только в поддерживаемых формат заллах
model.movie_hall_formats = py.ConstraintList()
for m in movie_names:
	allowed_h = movie_halls[m]
	for h in hall_names:
		if not (h in allowed_h):
			model.movie_hall_formats.add(sum(model.x[h, m, t] for t in period) == 0)
			
"""
Непересечение сеансов можно задать через линейные логические неравенства с использованием NBIG,
а можно через интервалы и ограничение на них NoOverLap
"""		
# в каждом зале в каждый момент времени начинается не больше одного сеанса
model.shows_not_intersect = py.ConstraintList()
for h in hall_names:
	for t in period:
		model.shows_not_intersect.add(sum(model.x[h, m, t] for m in movie_names) <= 1)
# сеансы не пересекаются
for h in hall_names:
	for m in movie_names:
		d = movies[m].len
		for tstart in range(T):
			tend = min(tstart + d, T - 1)
			#print(d, tstart, tend)
			model.shows_not_intersect.add(NBIG * model.x[h, m, tstart] + sum(model.x[h, m1, t] for t in range(tstart + 1, tend + 1) for m1 in movie_names) <= NBIG)

############################################################################
# Objective
############################################################################

model.sales = py.Objective(
	expr=sum(sales[m][t] * model.x[h, m, t] for h in hall_names for m in movie_names for t in period),
	sense=py.maximize
)

############################################################################
# Solve
############################################################################
solver = py.SolverFactory('cbc')
start_time = time()
result = solver.solve(model)
end_time = time()
status = result.Solver()["Termination condition"]
############################################################################
# Solution
############################################################################
if status == py.TerminationCondition.optimal or status == py.TerminationCondition.feasible:
	for h in hall_names:
		for t in period:
			for m in movie_names:
				if model.x[h, m, t]() > 0.5:
					print("hall {0}, time {1}, movie {2}, value {3}".format(h, t, m, model.x[h, m, t]()))
else:
	print('No solution found.')

print(model.sales())
print(end_time - start_time)
from sys import executable
from time import time
import numpy as np
from dataclasses import dataclass
import pyscipopt as scip
import sys; print('Python %s on %s' % (sys.version, sys.platform))

############################################################################
# Data
############################################################################

NBIG = 100
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
	supported_formats: list     # поддерживаемые форматы
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

model = scip.Model("SCIP Schedule")

############################################################################
# Decision vars
############################################################################

# vtype: type of the variable: 'C' continuous, 'I' integer, 'B' binary, and 'M' implicit integer
model_x = [[[model.addVar(name=f"x[{h},{m},{t}]", vtype="B") for t in period] for m in movie_names] for h in hall_names]

############################################################################
# Constraints
############################################################################

# Максиммальное и минимальное число фильмов
constr_show_ranges = []
for (ixm, m) in enumerate(movie_names):
	constr_show_ranges.append(
		model.addCons(sum(model_x[ixh][ixm][t] for (ixh, h) in enumerate(hall_names) for t in period) >= movies[m].min)
	)
	constr_show_ranges.append(
		model.addCons(sum(model_x[ixh][ixm][t] for (ixh, h) in enumerate(hall_names) for t in period) <= movies[m].max)
	)


# Фильмы только в поддерживаемых формат заллах
constr_movie_hall_formats = []
for (ixm, m) in enumerate(movie_names):
	allowed_h = movie_halls[m]
	for (ixh, h) in enumerate(hall_names):
		if not (h in allowed_h):
			constr_movie_hall_formats.append(model.addCons(sum(model_x[ixh][ixm][t] for t in period) == 0))
			
"""
Непересечение сеансов можно задать через линейные логические неравенства с использованием NBIG,
а можно через клики
"""		
# в каждом зале в каждый момент времени начинается не больше одного сеанса
constr_shows_not_intersect = []
for (ixh, h) in enumerate(hall_names):
	for t in period:
		constr_shows_not_intersect.append(model.addCons(sum(model_x[ixh][ixm][t] for (ixm, m) in enumerate(movie_names)) <= 1))

# сеансы не пересекаются
for (ixh, h) in enumerate(hall_names):
	for (ixm, m) in enumerate(movie_names):
		d = movies[m].len
		# Число фильмов в зале за промежуток времени d не должно превышать d / минимальная длительность фильма
		big_m = int(np.ceil(d / min([movies[mov].len for mov in movies.keys()])))
		big_m = NBIG
		for tstart in range(T):
			tend = min(tstart + d, T - 1)
			# print(d, tstart, tend)
			# BIG M
			constr_shows_not_intersect.append(
				model.addCons(
					big_m * model_x[ixh][ixm][tstart] + sum(model_x[ixh][ixm1][t]
					for t in range(tstart + 1, tend + 1) for (ixm1, m1) in enumerate(movie_names)) <= big_m)
			)
			# Cliques
			# for t in range(tstart + 1, tend + 1):
			# 	constr_shows_not_intersect.add(model.addCons(model_x[ixh][ixm][tstart] + sum(model_x[ixh][ixm1][t] for (ixm1, m1) in enumerate(movie_names)) <= 1))

############################################################################
# Objective
############################################################################

model.setObjective(sum(sales[m][t] * model_x[ixh][ixm][t] for (ixh, h) in enumerate(hall_names) for (ixm, m) in enumerate(movie_names) for t in period), sense='maximize')

model.writeProblem(filename="scip_problem_init.lp", trans=False, genericnames=False)
############################################################################
# Solve
############################################################################
sol_x = [[[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
 [[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
 [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]]

# sol_1 = model.createPartialSol()
sol_1 = model.createSol()
for (ixh, h) in enumerate(hall_names):
	for t in period:
		for (ixm, m) in enumerate(movie_names):
			model.setSolVal(sol_1, model_x[ixh][ixm][t], int(sol_x[ixh][ixm][t]))

# Проверяем и добавляем решение
accepted = model.addSol(sol_1, free=False)
print(f"Warm start solution accepted: {accepted}")

# model.getParam("constraints/setppc/cliquelifting")
# model.getParam("display/verblevel")
# model.setParam("display/verblevel", 5)
model.setParam('limits/time', 600)
# model.setParam('constraints/setppc/cliquelifting', True)
model.setPresolve(scip.SCIP_PARAMSETTING.OFF)
model.setParam("limits/gap", 10)

# model.hideOutput()
model.setLogfile("scip_log.txt")
start_time = time()
model.optimize()
sol = model.getBestSol()
end_time = time()

status = model.getStatus()
print(status)

model.writeProblem(filename="scip_problem.lp", trans=True, genericnames=False)

############################################################################
# Solution
############################################################################
if status in ("optimal", "gaplimit"):
	for (ixh, h) in enumerate(hall_names):
		for t in period:
			for (ixm, m) in enumerate(movie_names):
				if sol[model_x[ixh][ixm][t]] > 0:
					print("hall: {0}, start: {1}, movie: {2}, duration: {3}".format(h, t, m, movies[m].len))
else:
	print('No solution found.')

print(model.getObjVal())
print(end_time - start_time)

# sol_x = [[[int(model.getVal(model_x[ixh][ixm][t])) for t in period] for (ixm, m) in enumerate(movie_names)] for (ixh, h) in enumerate(hall_names)]


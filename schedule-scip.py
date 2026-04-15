from sys import executable
from time import time
import numpy as np
from dataclasses import dataclass
import pyscipopt as scip
import sys; print('Python %s on %s' % (sys.version, sys.platform))

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
	supported_formats: list[str]     # поддерживаемые форматы
	max_shows: int                              # максимальное число сеансов
	def __post_init__(self):
		assert(type(self.supported_formats) is list)
		assert(all([f in movie_formats for f in self.supported_formats]))

halls = {
	"hall_1": Hall(supported_formats=["A"], max_shows=15),
	"hall_2": Hall(supported_formats=["A"], max_shows=12),
	"hall_3": Hall(supported_formats=["A", "B"], max_shows=10),
}
hall_names = list(halls.keys())

movie_halls = {movie : [hall for hall in halls if movies[movie].format in halls[hall].supported_formats] for movie in movies}
hall_movies = {hall : [movie for movie in movies if movies[movie].format in halls[hall].supported_formats] for hall in halls}

# Максимальное число сеансов в зале
hall_max_shows = {}
for h in halls.keys():
	hall_max_shows[h] = min(
		halls[h].max_shows, 
		T // min(movies[m].len for m in movies.keys() if h in movie_halls[m])
	)

# Максимальное число сеансов фильма в зале
hall_movie_max_shows = {}
for h in halls.keys():
	for m in movies.keys():
		hall_movie_max_shows[h, m] = 0 if h not in movie_halls[m] else min(movies[m].max, hall_max_shows[h])

np.random.seed(1)
sales = {m: [np.random.randint(low=1, high=10) for t in range(T)] for m in movies}

############################################################################
# Модель
############################################################################

model = scip.Model("SCIP Schedule")

############################################################################
# Decision vars + Constraints
############################################################################

# vtype: type of the variable: 'C' continuous, 'I' integer, 'B' binary, and 'M' implicit integer
# Число сеансов данного фильма во всех залах
movie_count = {
	m: model.addVar(
		name=f"{m}_count", 
		lb=movies[m].min, 
		ub=movies[m].max,
		vtype="I"
	) for m in movies.keys()
}

# Число сеансов всех фильмов в данном зале
hall_count = {
	h: model.addVar(
		name=f"{h}_count", 
		lb=0, 
		ub=hall_max_shows[h],
		vtype="I"
	) for h in halls.keys()
}

# Число сеансов данного фильма в данном зале
movie_hall_count = {
	(h, m): model.addVar(
		name=f"{h}_{m}_count", 
		lb=0, 
		ub=hall_movie_max_shows[h, m], # Фильмы только в поддерживаемых его формат залах
		vtype="I"
	) for m in movies.keys() for h in halls.keys()
}

# Согласовываем количество сеансов фильмов в залах
for m in movies.keys():
	model.addCons(sum(movie_hall_count[h, m] for h in halls.keys()) == movie_count[m])
for h in halls.keys():
	model.addCons(sum(movie_hall_count[h, m] for m in movies.keys()) == hall_count[h])

# Фильмы только в поддерживаемых его формат залах
for m in movies:
	allowed_h = movie_halls[m]
	for h in halls:
		if not (h in allowed_h):
			model.addCons(movie_hall_count[h, m] == 0)

# Начинается ли сеанс данного фильма в данном зале в определенный момент времени
x = {(h, m, t): model.addVar(name=f"x[{h}, {m}, {t}]", vtype="B") for h in halls.keys() for m in movies.keys() for t in period}

# Согласовываем назначение сеансов с числом сеансов каждого фильма в каждом зале
for h in halls.keys():
	for m in movies.keys():
		model.addCons(sum(x[h, m, t] for t in period) == movie_hall_count[h, m])

# в каждом зале в каждый момент времени начинается не больше одного сеанса
for h in halls.keys():
	for t in period:
		model.addCons(sum(x[h, m, t] for m in movies.keys()) <= 1)

# сеансы не пересекаются (ограничение через BIG_M)
for h in halls.keys():
	for m in movies.keys():
		d = movies[m].len
		for tstart in range(T):
			tend = tstart + np.min([T - tstart - 1, d])
			#print(d, tstart, tend)
			model.addCons(d * x[h, m, tstart] + sum(x[h, m1, t] for t in range(tstart + 1, tend + 1) for m1 in movies.keys()) <= d)

############################################################################
# Objective
############################################################################

model.setObjective(sum(sales[m][t] * x[h, m, t] for h in halls.keys() for m in movies.keys() for t in period), sense='maximize')

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
# sol_1 = model.createSol()
# for (ixh, h) in enumerate(hall_names):
# 	for t in period:
# 		for (ixm, m) in enumerate(movie_names):
# 			model.setSolVal(sol_1, x[h, m, t], int(sol_x[ixh][ixm][t]))

# # Проверяем и добавляем решение
# accepted = model.addSol(sol_1, free=False)
# if accepted:
# 	print(f"Warm start solution was accepted: {accepted}")
# 	print(model.getSolObjVal(sol_1))
# else:
# 	print(f"Warm start solution was not accepted: {accepted}")

# model.getParam("constraints/setppc/cliquelifting")
# model.getParam("display/verblevel")
model.setParam("display/verblevel", 1)
model.setParam('limits/time', 600)
# model.setParam('constraints/setppc/cliquelifting', True)
# model.setPresolve(scip.SCIP_PARAMSETTING.OFF)
# model.setParam("limits/gap", 10)

# model.hideOutput()
# model.setLogfile("scip_log.txt")
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
				if sol[x[h, m, t]] > 0:
					print(f"hall: {h}, start: {t}, movie: {m}, duration: {movies[m].len}")
else:
	print('No solution found.')

print(model.getSolObjVal(sol))
print(end_time - start_time)


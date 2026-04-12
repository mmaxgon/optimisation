"""
https://developers.google.com/optimization/reference/python/sat/python/cp_model
https://github.com/google/or-tools/blob/stable/ortools/sat/samples/
"""
from time import time
import numpy as np
from ortools.sat.python import cp_model

############################################################################
# Data
############################################################################

# Число 
T = 24
period = range(T)

# Форматы фильмов
movie_formats = set(["A", "B"])

movies = {
	"movie_1" : {"len" : 5, "min": 3, "max": 5, "format": "A"},	
	"movie_2" : {"len" : 3, "min": 4, "max": 10, "format": "A"},	
	"movie_3" : {"len" : 7, "min": 2, "max": 4, "format": "B"}
}
movie_names = list(movies.keys())

# Залы
halls = {
	"hall_1": {"supported_formats": ["A"], "max_shows": 15},
	"hall_2": {"supported_formats": ["A"], "max_shows": 12},
	"hall_3": {"supported_formats": ["A", "B"], "max_shows": 10},
}
hall_names = list(halls.keys())

# в каких заллах можно показать фильм
movie_halls = {m : [h for h in halls if movies[m]["format"] in halls[h]["supported_formats"]] for m in movies}
# какие фильмы можно показать в зале
hall_movies = {h : [m for m in movies if movies[m]["format"] in halls[h]["supported_formats"]] for h in halls}

# Максимальное число сеансов в зале
hall_max_shows = {}
for h in halls:
	hall_max_shows[h] = min(
		halls[h]["max_shows"], 
		T // min(movies[m]["len"] for m in movies if h in movie_halls[m])
	)

np.random.seed(1)
sales = {m: [np.random.randint(low=1, high=10) for t in range(T)] for m in movies}

############################################################################
# Модель
############################################################################

model = cp_model.CpModel()

############################################################################
# Decision vars + Constraints
############################################################################

# Число сеансов каждого фильма во всех залах
movie_count = {
	m: model.NewIntVar(
		name=f"{m}_count", 
		lb=movies[m]["min"], 
		ub=movies[m]["max"]
	) for m in movies
}

# Число сеансов каждого фильма в каждом зале
movie_hall_count = {
	(h, m): model.NewIntVar(
		name=f"{h}_{m}_count", 
		lb=0, 
		ub=0 if h not in movie_halls[m] else min(movies[m]["max"], hall_max_shows[h]) # Фильмы только в поддерживаемых его формат залах
	) for m in movies for h in halls
}

# Согласовываем количество сеансов фильмов в залах
for m in movies:
	model.add(sum(movie_hall_count[h, m] for h in halls) == movie_count[m])

# Фильмы только в поддерживаемых его формат залах
for m in movies:
	allowed_h = movie_halls[m]
	for h in halls:
		if not (h in allowed_h):
			model.add(movie_hall_count[h, m] == 0)

# Показывается ли фильм m в зале h i-ым сеансом?
x = {(h, m, i): model.NewBoolVar("x[{}, {}, {}]".format(h, m, i)) for i in range(hall_max_shows[h]) for m in movies for h in halls}

# В каждом зале на каждом сеансе показывают не более одного фильма
for h in halls:
	for i in range(hall_max_shows[h]):
		model.add(sum(x[h, m, i] for m in movies) <= 1)

# Борьба с симметрией: пустые сеансы в каждом зале идут в конце
for h in halls:
	for i in range(1, hall_max_shows[h]):
		model.add(sum(x[h, m, i] for m in movies) <= sum(x[h, m, i-1] for m in movies))

# Соблюдаем число сеансов каждого фильма в каждом зале
for h in halls:
	for m in movies:
		model.add(sum(x[h, m, i] for i in range(hall_max_shows[h])) == movie_hall_count[h, m])

# Время начала i-ого сеанса каждого фильма в каждом зале
start_movie_hall = {
	(h, m, i): model.NewIntVar(name=f"начало сеанса {i} в зале {h} фильма {m}", lb=0, ub=T-1) 
	for i in range(hall_max_shows[h]) 
	for m in movies 
	for h in halls
}

# Время окончания i-ого сеанса каждого фильма в каждом зале
end_movie_hall = {
	(h, m, i): model.NewIntVar(name=f"конец сеанса {i} в зале {h} фильма {m}", lb=0, ub=T-1+max(movies[m]["len"] for m in movies if h in movie_halls[m]))
	for i in range(hall_max_shows[h]) 
	for m in movies 
	for h in halls
}

# Согласовываем начало и конец сеансов
for h in halls:
	for m in movies:
		for i in range(hall_max_shows[h]):
			model.add(end_movie_hall[h, m, i] == start_movie_hall[h, m, i] + movies[m]["len"] + 1).OnlyEnforceIf(x[h, m, i])

# У сеансов строго возрастающее время начала
for h in halls:
	for i in range(1, hall_max_shows[h]):
		for m1 in movies:
			for m2 in movies:
				model.add(start_movie_hall[h, m1, i] >= start_movie_hall[h, m2, i-1]).OnlyEnforceIf(x[h, m1, i]) #.OnlyEnforceIf(x[h, m2, i-1])

# Интервалы сеансов в каждом зале
shows = {
	(h, m, i): 
	model.new_optional_interval_var(
		start=start_movie_hall[h, m, i],
		size=movies[m]["len"] + 1,
		end=end_movie_hall[h, m, i],
		is_present=x[h, m, i],
		name="show[{}, {}, {}]".format(h, m, i)
	)	
	for i in range(hall_max_shows[h]) 
	for m in movies 
	for h in halls
}

# сеансы не пересекаются (ограничение через интервалы)
for h in halls:
	model.add_no_overlap([shows[h, m, i] for i in range(hall_max_shows[h]) for m in movies])

# Коэффициенты при целевой функции, индексированные номером сеанса, а не временным индексом
sales_x = {
	(h, m, i): model.NewIntVar(name=f"sales_x[{h}, {m}, {i}]", lb=0, ub=10)
	for i in range(hall_max_shows[h])
	for m in movies
	for h in halls
}

# Переводим номер сеанса во временной индекс
for h in halls:
	for m in movies:
		for i in range(hall_max_shows[h]):
			# variables[index] == target
			# sales[m][start_movie_hall[h, m, i]] == sales_x[h, m, i]
			model.add_element(
				index=start_movie_hall[h, m, i], 
				expressions=sales[m], 
				target=sales_x[h, m, i]
			).OnlyEnforceIf(x[h, m, i])
			model.add(sales_x[h, m, i] == 0).OnlyEnforceIf(x[h, m, i].Not())

############################################################################
# Objective
############################################################################
model.Maximize(sum(sales_x[h, m, i] for h in halls for m in movies for i in range(hall_max_shows[h])))
# model.Maximize(
# 	sum(1 * x[h, m, i] 
# 		for i in range(hall_max_shows[h])
# 		for m in movies
# 		for h in halls
# 	)
# )
# model.Maximize(sum(movie_count[m] for m in movies))

############################################################################
# Solve
############################################################################
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60.0
start_time = time()
status = solver.Solve(model)
end_time = time()

############################################################################
# Solution
############################################################################
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
	for ix_h, h in enumerate(halls):
		for i in range(hall_max_shows[h]):
			for ix_m, m in enumerate(movies):
				if solver.BooleanValue(x[h, m, i]):
					print(f"{i}: hall {h}, time {solver.Value(start_movie_hall[h, m, i])}, movie {m}")
else:
	print('No solution found.')

print(solver.StatusName(status))
print(solver.ObjectiveValue())
print(end_time - start_time)

# {ix: solver.Value(movie_count[ix]) for ix in movie_count.keys()}
# {ix: solver.Value(movie_hall_count[ix]) for ix in movie_hall_count.keys()}

# m = "movie_2"
# h = "hall_1"
# i = 1
# t = solver.Value(start_movie_hall[h, m, i])
# print(t)
# solver.Value(sales_x[h, m, i])
# sales[m][t]
# sales[m][t] == solver.Value(sales_x[h, m, i])

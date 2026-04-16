from time import time
import random
from ortools.sat.python import cp_model
import pyscipopt as scip
from schedule_data import *

#############################################################
# Решение 1: сколько сеансов каждого фильма идёт в каждом зале
#############################################################

def get_movie_hall_count(movie_hall_count_rand):
    """
    Cколько сеансов каждого фильма идёт в каждом зале
    """
    model = scip.Model("Cколько сеансов каждого фильма идёт в каждом зале")

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
        # Число сеансов фильма m во всех залах
        model.addCons(sum(movie_hall_count[h, m] for h in halls) == movie_count[m])
    for h in halls:
        # Число сеансов всех фильмов в зале h
        model.addCons(sum(movie_hall_count[h, m] for m in movies) == hall_count[h])
        # Ограничение на длительность всех сеансов в зале h
        model.addCons(sum((movies[m].len + 1) * movie_hall_count[h, m] for m in movies) <= T + min(movies[m].len for m in hall_movies[h]))

    # Минимизируем отклонения числа сеансов с "догадкой"
    mu = {(h, m): model.addVar(name=f"mu[{h}, {m}]", vtype="C") for m in movies for h in halls}
    for h in halls:
        for m in movies:
            model.addCons(movie_hall_count[h, m] - movie_hall_count_rand[h, m] <= mu[h, m])
            model.addCons(movie_hall_count_rand[h, m] - movie_hall_count[h, m] <= mu[h, m])
    model.setObjective(sum(mu[h, m ] for h in halls for m in movies), sense='minimize')

    model.setParam("display/verblevel", 1)
    model.setParam('limits/time', 10)

    start_time = time()
    model.optimize()
    sol = model.getBestSol()
    end_time = time()
    status = model.getStatus()

    if status in ("optimal", "gaplimit"):
        movie_hall_count_res = {(h, m): int(round(sol[movie_hall_count[h, m]])) for h in halls for m in movies}
    else:
        movie_hall_count_res = None
        
    return (movie_hall_count_res, end_time-start_time)

#############################################################
# Решение 2: Последовательность фильмов в зале
#############################################################

def get_movie_hall_seq(movie_hall_count):
    halls_show_count = {h: sum(movie_hall_count[h, m] for m in movies) for h in halls}

    start_time = time()
    res = {}
    for h in halls:
        elements = {m: movie_hall_count[h, m] for m in movies if movie_hall_count[h, m] > 0}
        items = []
        for item, count in elements.items():
            items.extend([item] * count)
        # Перемешиваем
        random.shuffle(items)
        for i in range(halls_show_count[h]):
            res[h, i] = items[i]
    end_time = time()

    # model = scip.Model("Последовательность фильмов в зале")

    # # Идёт ли в зале h фильм m i-ым сеансом?
    # x = {(h, m, i): model.addVar(name=f"x[{h}, {m}, {i}]", vtype="B") for h in halls for m in movies for i in range(halls_show_count[h])}

    # # В каждом зале на каждом сеансе показывают ровно один фильм
    # for h in halls:
    #     for i in range(halls_show_count[h]):
    #         model.addCons(sum(x[h, m, i] for m in movies) == 1)

    # # Соблюдаем число сеансов каждого фильма в каждом зале
    # for h in halls:
    #     for m in movies:
    #         model.addCons(sum(x[h, m, i] for i in range(halls_show_count[h])) == movie_hall_count[h, m])

    # model.setParam("display/verblevel", 1)
    # model.setParam('limits/time', 10)

    # start_time = time()
    # model.optimize()
    # sol = model.getBestSol()
    # end_time = time()
    # status = model.getStatus()

    # res = {}
    # if status in ("optimal", "gaplimit"):
    #     for h in halls:
    #         for i in range(halls_show_count[h]):
    #             for m in movies:
    #                 if sol[x[h, m, i]]:
    #                     res[(h, i)] = m
    # else:
    #     res = None
	
    return (halls_show_count, res, end_time-start_time)

#############################################################
# Решение 3: Время сеансов
#############################################################

def get_schedule(halls_show_count, movie_hall_seq):
    model = cp_model.CpModel()

    # Время начала i-ого сеанса каждого фильма в каждом зале
    start_movie_hall = {
        (h, i): model.new_int_var(name=f"Начало сеанса {i} в зале {h} фильма {movie_hall_seq[h, i]}", lb=0, ub=T-1) 
        for (h, i) in movie_hall_seq.keys() 
    }

    # Время окончания i-ого сеанса каждого фильма в каждом зале
    end_movie_hall = {
        (h, i): model.new_int_var(name=f"Конец сеанса {i} в зале {h} фильма {movie_hall_seq[h, i]}", lb=0, ub=T-1+max(movies[m].len for m in hall_movies[h]))
        for (h, i) in movie_hall_seq.keys() 
    }

    # Согласовываем начало и конец сеансов
    for (h, i) in movie_hall_seq.keys():
        model.add(end_movie_hall[h, i] == start_movie_hall[h, i] + movies[movie_hall_seq[h, i]].len)
                
    # У сеансов строго возрастающее время начала
    for h in halls:
        for i in range(1, halls_show_count[h]):
            model.add(start_movie_hall[h, i] >= end_movie_hall[h, i-1] + 1)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    start_time = time()
    status = solver.Solve(model)
    end_time = time()

    res = {}
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for h in halls:
            for i in range(halls_show_count[h]):
                res[(h, i)] = {
                    "movie": movie_hall_seq[h, i], 
                    "start": solver.value(start_movie_hall[h, i]), 
                    "end": solver.value(end_movie_hall[h, i])
                }
    else:
        res = None

    return (res, end_time-start_time)

#########################################################################
# Функция цели
#########################################################################
def get_goal(schedule):
    return sum(sales[schedule[ix]["movie"]][schedule[ix]["start"]] for ix in schedule.keys())


#########################################################################

movie_hall_count_rand = {(h, m): 0 if hall_movie_max_shows[h, m] == 0 else np.random.randint(low=0, high=hall_movie_max_shows[h, m]) for m in movies for h in halls}
(movie_hall_count, duration) = get_movie_hall_count(movie_hall_count_rand)
print(f"{movie_hall_count} \n {duration}")

(halls_show_count, movie_hall_seq, duration) = get_movie_hall_seq(movie_hall_count)
print(f"{movie_hall_seq} \n {duration}")

(schedule, duration) = get_schedule(halls_show_count, movie_hall_seq)
print(f"{schedule} \n {duration}, \n {get_goal(schedule)}")


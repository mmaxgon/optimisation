from time import time
import numpy as np
import pyscipopt as scip
from schedule_data import *

############################################################################
# Модель
############################################################################

# Создаём модель с указанием, что это задача максимизации
model = scip.Model("SCIP Schedule")
model.setObjectiveSense('maximize')  # Явно задаём направление оптимизации

############################################################################
# Переменные решения + Ограничения
############################################################################

# Число сеансов каждого фильма во всех залах
movie_count = {
    m: model.addVar(
        name=f"{m}_count",
        lb=movies[m].min,
        ub=movies[m].max,
        vtype="I"  # Целочисленная переменная
    ) for m in movies
}

# Число сеансов всех фильмов в каждом зале
hall_count = {
    h: model.addVar(
        name=f"{h}_count",
        lb=0,
        ub=hall_max_shows[h],
        vtype="I"
    ) for h in halls
}

# Число сеансов каждого фильма в каждом зале
movie_hall_count = {
    (h, m): model.addVar(
        name=f"{h}_{m}_count",
        lb=0,
        ub=hall_movie_max_shows[h, m],
        vtype="I"
    ) for h in halls for m in movies  
}

# Связываем общее число сеансов фильма по залам
for m in movies:
    model.addCons(sum(movie_hall_count[h, m] for h in halls) == movie_count[m])

# Связываем общее число сеансов в зале по фильмам
for h in halls:
    model.addCons(sum(movie_hall_count[h, m] for m in movies) == hall_count[h])

# Булевы переменные: начинается ли сеанс (зал, фильм) в момент времени t
x = {
    (h, m, t): model.addVar(name=f"x[{h},{m},{t}]", vtype="B")  # Упрощено имя переменной
    for h in halls
    for m in hall_movies[h]
    for t in valid_starts[m]
}

# Связываем булевы переменные с количеством сеансов
for h in halls:
    for m in movies:
        if h in movie_halls[m]:
            model.addCons(sum(x[h, m, t] for t in valid_starts[m]) == movie_hall_count[h, m])
        else:
            model.addCons(movie_hall_count[h, m] == 0)

# В каждом зале в каждый момент времени начинается не более одного сеанса
for h in halls:
    for t in period:
        starts_at_t = [x[h, m, t] for m in hall_movies[h] if (h, m, t) in x]
        if starts_at_t:
            model.addCons(sum(starts_at_t) <= 1)

# Ограничение на непересечение сеансов (альтернативный подход через интервалы)
# Вместо BIG-M используем логическое выражение: если сеанс (m1,t1) и (m2,t2) перекрываются — нельзя одновременно
# Однако для производительности оставим упрощённую версию, но улучшим читаемость
for h in halls:
    for m in hall_movies[h]:
        d = movies[m].len
        for t_start in valid_starts[m]:
            # Все другие возможные сеансы, которые пересекаются по времени
            conflicting = [
                x[h, m_other, t_other]
                for m_other in hall_movies[h]
                for t_other in valid_starts[m_other]
                if (h, m_other, t_other) in x and (t_other <= t_start < t_other + movies[m_other].len + 1 or
                                                  t_start <= t_other < t_start + d + 1)
            ]
            # Если текущий сеанс назначен — ни один из конфликтующих быть не должен
            if conflicting:
                model.addCons(scip.quicksum(conflicting) <= len(conflicting) * (1 - x[h, m, t_start]))

############################################################################
# Целевая функция
############################################################################

# Максимизация суммарных продаж
objective = scip.quicksum(sales[m][t] * x[h, m, t] for (h, m, t) in x.keys())
model.setObjective(objective, sense='maximize')

############################################################################
# Настройки решателя
############################################################################

# Подавляем подробный вывод, если не нужен
model.setParam("display/verblevel", 1)  # Можно изменить на 0 для тишины
model.setParam('limits/time', 600)      # Лимит времени — 10 минут
model.setParam('limits/gap', 0.01)      # Останов при достижении разрыва < 1%
model.setHeuristics(scip.SCIP_PARAMSETTING.FAST)  # Быстрые эвристики для ускорения

############################################################################
# Решение
############################################################################

start_time = time()
model.optimize()
end_time = time()

status = model.getStatus()
print(f"Статус решения: {status}")
print(f"Время выполнения: {end_time - start_time:.2f} сек")

############################################################################
# Вывод результата
############################################################################

sol = model.getBestSol()
if sol is not None and status in ("optimal", "gaplimit", "timelimit"):
    total_sales = model.getObjVal()  # Получаем значение целевой функции
    print(f"\nНайдено решение. Суммарные продажи: {total_sales:.0f}\n")
    
    # Сортируем расписание по времени начала
    schedule_list = [
        (h, t, m, movies[m].len)
        for (h, m, t) in x.keys()
        if sol[x[h, m, t]] > 0.5
    ]
    schedule_list.sort(key=lambda item: item[1])  # Сортировка по времени
    
    for h, t, m, d in schedule_list:
        print(f"Зал: {h}, Фильм: {m}, Начало: {t}, Длительность: {d}")
else:
    print("Решение не найдено.")

print(f"Общее время: {end_time - start_time:.2f} сек")
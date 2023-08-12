import numpy as np
import scipy.linalg as la
import scipy.optimize as opt

# Длина доски
L = 10
# Длины брусков
l_lens = [1, 3, 5]
# Потребности в брусках
l_demands = np.array([19, 9, 7])
# Число разных брусков
l_count = len(l_lens)

# Первоначальный набор шаблонов нарезки (тупой: шаблон состоит из одного бруска)
patterns = np.eye(l_count)

"""
Решаем мастер-задачу: 
сколько каждого шаблона из текущего набора надо использовать, чтобы 
- удовлетворить всем потребностям и 
- уменьшить суммарное число используемых досок
"""
def solve_master_problem(integrality=0):
	# число столбцов (шаблонов):
	m = patterns.shape[1]
	# функция цели - минимизировать число используемых досок (шаблонов)
	c = np.ones(patterns.shape[1])
	# ограничения - число досок каждого типа по всем используемым паттернам >= demand
	sol_master = opt.linprog(c=c, A_ub=-patterns, b_ub=-l_demands, method="highs", integrality=integrality)
	for i in range(m):
		print(f"Шаблон: {str(patterns[:, i])} используем {sol_master.x[i]} раз")
	# Проверка правильности расчёта двойственных переменных
	# sol_dual = opt.linprog(c=-l_demands, A_ub=patterns.T, b_ub=c)
	# print(sol_dual.x)
	# print(sol_master.ineqlin.marginals)
	return sol_master

"""
Находим новый паттерн:
- суммарные длины всех брусков в нём не превышают длины доски
- новый паттерн должен максимально потенциально уменьшать ЦФ исходной задачи
"""
def solve_knapsack_problem(sol_master):
	duals = -sol_master.ineqlin.marginals
	sol_knapsack = opt.linprog(c=-duals, A_ub=[l_lens], b_ub=[L], integrality=1)
	print(f"Добавляем шаблон {str(sol_knapsack.x)}")
	return sol_knapsack

sol_master = solve_master_problem()

# Итерируем
for i in range(100):
	sol_knapsack = solve_knapsack_problem(sol_master)
	if 1 + sol_knapsack.fun < -1e-4:
		patterns = np.hstack([patterns, sol_knapsack.x.reshape(-1, 1)])
		sol_master = solve_master_problem()
	else:
		print("done!")
		break

# Получили паттерны - теперь решаем целочисленную задачу
sol = solve_master_problem(integrality=1)

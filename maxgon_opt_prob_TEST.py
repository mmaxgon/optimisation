import importlib

import maxgon_opt_prob as mg_opt
importlib.reload(mg_opt)

####################################################################################################
####################################################################################################
import copy
from time import time
import numpy as np

###############################################################################
'''
ЗАДАЧА

Decision variables:
var x{j in 1..3};

Bounds:
0 <= x[j] <= 10

Integer constraints:
x[2], x[3] integer

# Nonlinear Objective function to be minimized:
maximize obj:
        1000 - x[1]^2 - 2*x[2]^2 - x[3]^2 - x[1]*x[2] - x[1]*x[3];

# Linear Constraints:
s.t. c1: 8*x[1] + 14*x[2] + 7*x[3] - 56 = 0;

# Nonlinear Constraints
s.t. c2: x[1]^2 + x[2]^2 + x[3]^2 - 25 <= 0;
c3: x[2]^2 + x[3]^2 <= 12;	
'''

####################################################################################################
# Описание задачи
####################################################################################################
importlib.reload(mg_opt)
# Начальное значение
x0 = [0., 0., 0.]

# Переменные решения
decision_vars = mg_opt.dvars(
	n=3,                    # число переменных решения
	ix_int=[1, 2],          # индексы целочисленных
	ix_cont=[0],            # индексы непрерывных
	bounds=mg_opt.bounds(
		lb=[0, 0, 0],       # >= 0
		ub=[10, 10, 10]     # <= 10
	),
	x0=x0                   # начальное значение
)

# Целевые функции
# Нелинейная
def obj(x):
	return -(1000 - x[0]**2 - 2*x[1]**2 - x[2]**2 - x[0]*x[1] - x[0]*x[2])
# Линейная
def lin_obj(x):
	return 2*x[0] + 3*x[1] + 4*x[2]
# Объекты целевой функции
objective_fun = mg_opt.objective(
	n=3,                    # Число переменных решения
	if_linear=False,        # Признак линейности
	fun=obj,                # Ссылка на функцию
	lin_coeffs=None         # Коэффициенты для линейной функции
)
objective_lin_fun = mg_opt.objective(
	n=3,
	if_linear=True,
	fun=lin_obj,
	lin_coeffs=[2., 3., 4.])

# Линейные ограничения
lin_cons = mg_opt.linear_constraints(
		n=3,                # Число переменных решения
		m=1,                # Число линейных ограничений
		A=[                 # Матрица линейных ограничений
			[8, 14, 7]
		],
		bounds=mg_opt.bounds(   # Границы
			lb=[56],
			ub=[56]
		)
	)

# Нелинейные ограничения
def non_lin_cons_fun(x):
	return [
			x[0]**2 + x[1]**2 + x[2]**2 - 25,
			x[1]**2 + x[2]**2 - 12
		]
# Объект нелинейных ограничений
non_lin_cons = mg_opt.nonlinear_constraints(
		n=3,                            # Число переменных решения
		m=2,                            # Число нелинейных ограничений
		fun=non_lin_cons_fun,           # Нелинейная функция ограничений
		bounds=mg_opt.bounds(           # -inf <= ... <= 0
			lb=[-np.Inf, -np.Inf],
			ub=[0, 0]
		)
	)

# Объекты описания оптимизационной задачи:
opt_prob = mg_opt.optimization_problem(
	dvars=decision_vars,
	objective=objective_fun,
	linear_constraints=lin_cons,
	nonlinear_constraints=non_lin_cons
)
print(opt_prob.if_linear())
opt_prob_lin = mg_opt.optimization_problem(
	dvars=decision_vars,
	objective=objective_lin_fun,
	linear_constraints=lin_cons,
	nonlinear_constraints=None
)
print(opt_prob_lin.if_linear())

####################################################################################################
# Получение нижней границы решения как решение NLP-Задачи
####################################################################################################
# Нижняя граница как решение NLP-задачи
res_NLP_SCIPY = mg_opt.get_relaxed_solution(
	opt_prob, nlp_solver="SCIPY",
	options = {"verbose": 4, "maxiter": 100}
)
print(res_NLP_SCIPY)
res_NLP_NLOPT = mg_opt.get_relaxed_solution(
	opt_prob,
	nlp_solver="NLOPT",
	options={"xtol_abs": 1e-9}
)
print(res_NLP_NLOPT)
res_NLP_IPOPT = mg_opt.get_relaxed_solution(
	opt_prob,
	nlp_solver="IPOPT",
	options={"tol": 1e-4, "print_level": 4}
)
print(res_NLP_IPOPT)

####################################################################################################
# Получение допустимого решения как приближения непрерывного
####################################################################################################
init_feasible1 = mg_opt.get_feasible_solution1(opt_prob, res_NLP_SCIPY["x"], nlp_solver="SCIPY")
print(init_feasible1)

init_feasible2 = mg_opt.get_feasible_solution2(opt_prob, res_NLP_SCIPY["x"], nlp_solver="NLOPT")
print(init_feasible2)

####################################################################################################
# Получение решения методом Polyhedral Outer Approximation
####################################################################################################
# Нелинейная задача
nonlin_sol = mg_opt.get_POA_solution(
	opt_prob,
	if_nlp_lower_bound=True,
	if_refine=False,
	if_project=True,
	random_points_count=0,
	nlp_solver="NLOPT"
)
print(nonlin_sol)

# Линейная задача
lin_sol = mg_opt.get_POA_solution(
	opt_prob_lin,
	if_nlp_lower_bound=False,
	if_refine=False,
	if_project=False,
	random_points_count=0
)
print(lin_sol)

####################################################################################################
# Получение решения методом Branch and Bound
####################################################################################################
# Нелинейная задача
bb_nonlin_sol = mg_opt.get_BB_solution(opt_prob, int_eps=1e-4, nlp_solver="SCIPY")
print(bb_nonlin_sol)
bb_nonlin_sol = mg_opt.get_BB_solution(opt_prob, int_eps=1e-4, nlp_solver="IPOPT")
print(bb_nonlin_sol)
bb_nonlin_sol = mg_opt.get_BB_solution(opt_prob, int_eps=1e-4, nlp_solver="NLOPT")
print(bb_nonlin_sol)

# Линейная задача
bb_lin_sol = mg_opt.get_BB_solution(opt_prob_lin, int_eps=1e-4)
print(bb_lin_sol)

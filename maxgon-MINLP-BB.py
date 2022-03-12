##############################################################################
# MINLP with scipy and branch and bound
##############################################################################
import importlib

import maxgon_MINLP_POA as mg_minlp
importlib.reload(mg_minlp)

from collections import namedtuple
import numpy as np
import scipy.optimize as sp

##############################################################################
# Описание задачи
##############################################################################
# Начальное значение
x0 = [1.75, 3, 0]
# Переменные решения
decision_vars = mg_minlp.dvars(3, [1, 2], [0], mg_minlp.bounds([0, 0, 0], [10, 10, 10]), x0)

# Целевая функция
def obj(x):
	return -(1000 - x[0]**2 - 2*x[1]**2 - x[2]**2 - x[0]*x[1] - x[0]*x[2])
objective_fun = mg_minlp.objective(3, obj)

# Нелинейные ограничения
def non_lin_cons_fun(x):
	return [
			x[0]**2 + x[1]**2 + x[2]**2 - 25,
			x[1]**2 + x[2]**2 - 12
		]
non_lin_cons = mg_minlp.nonlinear_constraints(
		2,
		non_lin_cons_fun,
		mg_minlp.bounds([-np.Inf, -np.Inf], [0, 0])
	)

# Нелинейные ограничения для целиком целочисленной задачи
def non_lin_cons_fun_cp(x):
	return [
			x[0]**2 + x[1]**2 + x[2]**2 - 25,
			x[1]**2 + x[2]**2 - 13
		]
non_lin_cons_cp = mg_minlp.nonlinear_constraints(
		2,
		non_lin_cons_fun_cp,
		mg_minlp.bounds([-np.Inf, -np.Inf], [0, 0])
	)

# Линейные ограничения
lin_cons = mg_minlp.linear_constraints(
		1,
		[
			[8, 14, 7]
		],
		mg_minlp.bounds([56], [56])
	)

opt_prob = mg_minlp.optimization_problem(decision_vars, objective_fun, lin_cons, non_lin_cons)
opt_prob_cp = mg_minlp.optimization_problem(decision_vars, objective_fun, lin_cons, non_lin_cons_cp)

##############################################################################
# Решение
##############################################################################
# Нижняя граница как решение NLP-задачи
res_NLP = mg_minlp.get_NLP_lower_bound(opt_prob)
print(res_NLP)
nlp_lower_bound = res_NLP["obj"]

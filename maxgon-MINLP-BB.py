##############################################################################
# MINLP with scipy and branch and bound
##############################################################################

from collections import namedtuple
import numpy as np
import scipy.optimize as sp

##############################################################################
# Объекты описания задачи
##############################################################################

# Переменные решения
dvars = namedtuple("dvars", ["n", "ix_int", "ix_cont", "x0"])
# Границы
bounds = namedtuple("bounds", ["lb", "ub"])
# Цель
objective = namedtuple("objective", ["n", "fun"])
# Линейные ограничения
linear_constraints = namedtuple("linear_constraints", ["m", "A", "bounds"])
# Нелинейные ограничения
nonlinear_constraints = namedtuple("nonlinear_constraints", ["m", "fun", "bounds"])
# Описание оптимизационной задачи
optimization_problem = namedtuple("optimization_problem", ["dvars", "bounds", "objective", "linear_constraints", "nonlinear_constraints"])

##############################################################################
# Абстрактное описание задачи
##############################################################################

# Начальное значение
x0 = [2]*3
# Переменные решения
decision_vars = dvars(3, [1, 2], [0], x0)
# Границы
var_bounds = bounds([0, 0, 0], [10, 10, 10])

# Целевая функция
def obj(x):
	return -(1000 - x[0]**2 - 2*x[1]**2 - x[2]**2 - x[0]*x[1] - x[0]*x[2])
objective_fun = objective(3, obj)

# Нелинейные ограничения
def non_lin_cons_fun(x):
	return [
			x[0]**2 + x[1]**2 + x[2]**2 - 25,
			x[1]**2 + x[2]**2 - 12
		]
non_lin_cons = nonlinear_constraints(
		2,
		non_lin_cons_fun,
		bounds([-np.Inf, -np.Inf], [0, 0])
	)

# Линейные ограничения
lin_cons = linear_constraints(
		1,
		[
			[8, 14, 7]
		],
		bounds([56], [56])
	)

opt_prob = optimization_problem(decision_vars, var_bounds, objective_fun, lin_cons, non_lin_cons)

##############################################################################
# Решение
##############################################################################

import numpy as np
import nlopt

def goal(x, grad):
	return (100.0*(x[1]-x[0]**2.0)**2.0 + (1-x[0])**2.0)

lb = [0, -0.5]
ub = [1.0, 2.0]

def cons_(x):
	return np.concatenate((
		(np.array([x[0]**2 + x[1], x[0]**2 - x[1]]) - 1),
		np.dot([[1, 2], [2, 1]], x) - [1, 1]
	))
	
def cons(result, x, grad):
	result[:] = cons_(x)
	return result

x0 = np.array([0.5, 0])

opt = nlopt.opt(nlopt.LN_COBYLA, x0.size)

opt.set_min_objective(goal)

opt.set_lower_bounds(lb)
opt.set_upper_bounds(ub)

opt.add_inequality_mconstraint(cons, [1e-8] * 4)

opt.set_xtol_rel(1e-4)

x = opt.optimize(x0)

minf = opt.last_optimum_value()
print("optimum at ", x[0], x[1])
print("minimum value = ", minf)
print("result code = ", opt.last_optimize_result())

x
goal(x, None)
cons_(x)
lb - x
x - ub

####################################################################
def objective(x):
	return -(1000 - x[0] ** 2 - 2 * x[1] ** 2 - x[2] ** 2 - x[0] * x[1] - x[0] * x[2])

def eq_constraints(x):
	return np.array([
		8 * x[0] + 14 * x[1] + 7 * x[2] - 56
	])

def ineq_constraints(x):
	return np.array([
		x[0]**2 + x[1]**2 + x[2]**2 - 25,
		x[1]**2 + x[2]**2 - 12
	])

def _objective(x, grad):
	return objective(x)

def _ineq_constraints(result, x, grad):
	result[:] = ineq_constraints(x)
	return result

def _eq_constraints(result, x, grad):
	result[:] = eq_constraints(x)
	return result

x0 = np.array([0., 0., 10.])

nl_opt = nlopt.opt(nlopt.LN_COBYLA, x0.size)
nl_opt.set_min_objective(_objective)

nl_opt.add_equality_mconstraint(_eq_constraints, [1e-8]*2)
nl_opt.add_inequality_mconstraint(_ineq_constraints, [1e-8]*2)
nl_opt.set_lower_bounds([0.]*3)
nl_opt.set_upper_bounds([10.]*3)

nl_opt.set_xtol_rel(1e-4)

x = nl_opt.optimize(x0)

minf = nl_opt.last_optimum_value()
print("optimum at ", x[0], x[1], x[2])
print("minimum value = ", minf)
print("result code = ", nl_opt.last_optimize_result())

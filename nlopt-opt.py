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


import numpy as np
import scipy as sp

from scipy import optimize
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint

def rosen(x):
#	The Rosenbrock function
	return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

x0 = np.array([0.5, 0])

bounds = Bounds([0, -0.5], [1.0, 2.0])
# 0 <= x0 <= 1, -0.5 <= x1 <= 2.0

linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])
# x0 + 2*x1 <= 1
# 1 <= 2*x0 + x1 <= 1

def cons_f(x):
	return [x[0]**2 + x[1], x[0]**2 - x[1]]

nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, [1, 1])
# x0**2 + x1 <= 1
# x0**2 - x1 <= 1

rosen(x0)
x0 <= bounds.ub
x0 >= bounds.lb

np.dot(linear_constraint.A, x0) <= linear_constraint.ub 
np.dot(linear_constraint.A, x0) >= linear_constraint.lb 

np.array(nonlinear_constraint.fun(x0)) <= nonlinear_constraint.ub
np.array(nonlinear_constraint.fun(x0)) >= nonlinear_constraint.lb

#############################################################################
# SLSQP, trust-constr
#############################################################################
# Local optimization
methods = [
	#,'Nelder-Mead' 
	#,'Powell' 
	#,'CG'
	#,'BFGS'
	#'Newton-CG', 
	#,'L-BFGS-B' 
	#,'TNC'
	#'COBYLA', 
	'SLSQP'
	,'trust-constr' 
	#'dogleg', 
	#'trust-ncg', 
	#'trust-exact', 
	#'trust-krylov'
]
results = dict()
for method in methods:
	res = minimize(
		fun=rosen,
		bounds=bounds,
		constraints=[linear_constraint, nonlinear_constraint],
		x0=x0,
		method=method,
		options={'verbose': 0}
	)
	results[method] = res
	print(res.x)
	print(method + ": " + str(res.fun))

#############################################################################
# COBYLA
#############################################################################

def cons_lin(x):
	return -np.concatenate((
				np.dot(linear_constraint.A, x) - linear_constraint.ub,
				linear_constraint.lb - np.dot(linear_constraint.A, x)
			))
cons_lin(x0)

def cons_nonlin(x):
	return -(np.array([x[0]**2 + x[1], x[0]**2 - x[1]]) - nonlinear_constraint.ub)
cons_nonlin(x0)

cons = (
	{'type': 'ineq', 'fun': cons_nonlin},
	{'type': 'ineq', 'fun': cons_lin},
)
results["COBYLA"] = minimize(
	fun=rosen,
	constraints=cons,
	x0=x0,
	method="COBYLA",
	options={'verbose': 0}
)
print(res.x)
print(rosen(res.x))

# Global optimization
#############################################################################
# SHGO
#############################################################################

bounds_shgo = [(0, 1), (-0.5, 2.0)]
# 0 <= x0 <= 1
# -0.5 <= x1 <= 2.0

# x0 + 2*x1 <= 1
# 1 <= 2*x0 + x1 <= 1
def cons_lin1(x):
	return -(x[0] + 2*x[1] - 1)
def cons_lin2(x):
	return 2*x[0] + x[1] - 1

def cons_nonlin1(x):
	return cons_nonlin(x)[0]
def cons_nonlin2(x):
	return cons_nonlin(x)[1]

cons = (
	{'type': 'ineq', 'fun': cons_nonlin1},
	{'type': 'ineq', 'fun': cons_nonlin2},
	{'type': 'ineq', 'fun': cons_lin1},
	{'type': 'eq', 'fun': cons_lin2},
)
	
results["shgo"] = optimize.shgo(
	func=rosen,
	bounds=bounds_shgo,
	constraints=cons,
	sampling_method='sobol',
	minimizer_kwargs={"method": "SLSQP"}
)		 

for key in results:
	print(key + ":" + str(results[key].fun))
	
#############################################################################
# Differential Evolution
#############################################################################

results["differential_evolution"] = optimize.differential_evolution(
	func=rosen,
	bounds=bounds,
	constraints=[linear_constraint, nonlinear_constraint],
	polish=True,
	x0=x0,
)
results["differential_evolution"].x
results["differential_evolution"].fun
for key in results:
	print(key + ":" + str(results[key].fun))

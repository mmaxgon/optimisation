import copy
from time import time
import numpy as np
import matplotlib.pyplot as plt

from gekko import GEKKO
import pyomo.environ as pe
import scipy.optimize as opt
import cyipopt
import nlopt
import rbfopt
from ortools.sat.python import cp_model
##############################################################################
N = 10
n = 4

def cons(x):
	return sum(x)
##############################################################################
# GEKKO / APOPT
m = GEKKO(remote = False)
m.options.SOLVER = 1
m.options.LINEAR = 0

x = [m.Var(value = 1, lb = 0, ub = N, integer = True, name = "x" + str(i)) for i in range(n)]
cons(x)

m.Equation(cons(x) == N)

def obj(x):
	return sum(m.if2(x[i] - 1e-6, 0, x[i] * m.log(x[i])) for i in range(n))

obj(x)

m.Obj(obj(x))

start_time = time()
m.solve(disp = True)
print(time() - start_time)

m.options.objfcnval

x = [x[i][0] for i in range(n)]
print(x)
##############################################################################
# PYOMO
model = pe.ConcreteModel()
model.x = pe.Var(range(n), domain = pe.NonNegativeIntegers, bounds = (0, N), initialize = 1)

def pe_cons(model):
	return sum(model.x[i] for i in range(n)) == N

model.cons = pe.Constraint(rule = pe_cons)

def pe_obj(model):
 	# return sum(pe.Expr_if(model.x[i]<=1e-6, 0, model.x[i] * pe.log(model.x[i])) for i in range(n))
 	return sum(model.x[i] * pe.log(model.x[i] + 1e-6) for i in range(n))

model.obj = pe.Objective(
	rule = pe_obj,
	sense = pe.minimize
)

# PYOMO / mindtpy
pe.SolverFactory('cbc').available()
pe.SolverFactory('ipopt').available()
pe.SolverFactory('mindtpy').available()

start_time = time()
results = pe.SolverFactory('mindtpy').solve(
	model, 
 	strategy = 'ECP',
 	# strategy = 'OA',
	time_limit = 3600,
	mip_solver = 'cbc', 
	nlp_solver = 'ipopt', 
	tee = False
) 
print(time() - start_time)
results.write()

x = [model.x[i]() for i in range(n)]
x

##############################################################################
# PYOMO / couenne
pe.SolverFactory('couenne').available()

sf = pe.SolverFactory('couenne')
# sf.options["ipopt.display_stats"] = "yes"
start_time = time()
results = sf.solve(
	model, 
	tee = False,
	keepfiles = False
) 
print(time() - start_time)

x = [model.x[i]() for i in range(n)]
x

##############################################################################
# PYOMO / bonmin
pe.SolverFactory('bonmin').available()

sf = pe.SolverFactory('bonmin')
# sf.options["bonmin.algorithm"] = "B-BB"
sf.options["bonmin.algorithm"] = "B-OA"
# sf.options["bonmin.algorithm"] = "B-QG"

start_time = time()
results = sf.solve(
	model, 
	tee = True
) 
print(time() - start_time)

x = [model.x[i]() for i in range(n)]
x

##############################################################################
# SCIPY / trust-constr
K = 1e4
def nip(x):
	return K * np.sum([np.minimum(x[i] - np.floor(x[i]), np.ceil(x[i]) - x[i]) ** 2 for i in range(n)])

class gini:
	def objective(self, x):
		return np.sum([0 if x[i] < 1e-6 else x[i] * np.log(x[i]) for i in range(n)]) + nip(x)

	def gradient(self, x):
		return opt.approx_fprime(x, self.objective, epsilon = 1e-8)
	
	def hess(self, x):
		return np.zeros((n, n))
	
	def constraints(self, x):
		return np.array(np.sum([x[i] for i in range(n)]))
	
	def jacobian(self, x):
		return opt.slsqp.approx_jacobian(x, self.constraints, epsilon = 1e-8)

x0 = [1] * n

gi = gini()
gi.objective(x0)
gi.gradient(x0)
gi.constraints(x0)
gi.jacobian(x0)

# sp_cons = opt.LinearConstraint(np.ones(n), [N], [N], keep_feasible = True)
sp_cons = opt.NonlinearConstraint(gi.constraints, [N], [N], keep_feasible = True)

start_time = time()
results = opt.minimize(
		fun = gi.objective, 
		bounds = [(0, N) for i in range(n)],
		constraints = [sp_cons],
		x0 = x0, 
		method = "trust-constr", 
		options = {'verbose': 2}
	)
print(time() - start_time)

results.x

results.constr_violation

gi.constraints(results.x)

#############################################################################
# NLOPT
x0 = [1] * n
x = x0

K = 1e1
def nip(x):
	return K * np.sum([np.minimum(x[i] - np.floor(x[i]), np.ceil(x[i]) - x[i]) ** 2 for i in range(n)])

def goal(x, grad):
	return np.sum([0 if x[i] < 1e-6 else x[i] * np.log(x[i]) for i in range(n)]) + nip(x)

# скалаярное ограничения
def cons(x, grad):
	result = np.sum([x[i] for i in range(n)]) - N
	return result

# векторные ограничения
def v_cons(result, x, grad):
	result[:] = [np.sum([x[i] for i in range(n)]) - N]
	return result

opt_m = nlopt.opt(nlopt.LN_COBYLA, n)
opt_m.set_min_objective(goal)
opt_m.set_lower_bounds([0] * n)
opt_m.set_upper_bounds([N] * n)
# opt_m.add_equality_constraint(cons, 1e-6)
opt_m.add_equality_mconstraint(v_cons, [1e-6])
opt_m.set_xtol_rel(1e-4)

start_time = time()
x = opt_m.optimize(x0)
print(time() - start_time)
x

#############################################################################
# IPOPT
K = 10
def ipopt_nip(x):
	return K * np.sum([np.minimum(x[i] - np.floor(x[i]), np.ceil(x[i]) - x[i]) ** 2 for i in range(n)])

class ipopt_gini():
	def objective(self, x):
 		return np.sum([0 if x[i] < 1e-6 else x[i] * np.log(x[i]) for i in range(n)]) + ipopt_nip(x)

	def gradient(self, x):
		return opt.approx_fprime(x, self.objective, epsilon = 1e-8)
 	
	def constraints(self, x):
		return np.array([np.sum([x[i] for i in range(n)]), ipopt_nip(x)])
	
	def jacobian(self, x):
		return opt.slsqp.approx_jacobian(x, self.constraints, epsilon = 1e-8)

x0 = [1] * n
x = x0

gi = ipopt_gini()

lb = [0] * n
ub = [N] * n
cl = [N, 0]
cu = [N, 5]

nlp = cyipopt.Problem(
	n = n,
	m = len(cl),
	problem_obj = ipopt_gini(),
	lb = lb,
	ub = ub,
	cl = cl,
	cu = cu,
)

# nlp.add_option('jac_c_constant', 'yes')
# nlp.add_option('mu_strategy', 'adaptive')
# nlp.add_option('tol', 1e-7)

start_time = time()
x, info = nlp.solve(x0)
print(time() - start_time)
x
info

gi.objective(x)
gi.objective(x0)

##############################################################################
# rbfopt

K = 1e1
N = 4

def constr_eq(x):
	return (sum(x) - 10) ** 2

def objective(x):
	return np.sum([0 if x[i] < 1e-6 else x[i] * np.log(x[i]) for i in range(n)]) + \
	K * constr_eq(x)

bb = rbfopt.RbfoptUserBlackBox(
	dimension = N, 
	var_lower = np.array([0] * N), 
	var_upper = np.array([10] * N),
	var_type = np.array(['I'] * N), 
	obj_funct = objective,
	obj_funct_noisy = None
)

settings = rbfopt.RbfoptSettings(
	minlp_solver_path = 'bonmin', 
	nlp_solver_path = 'ipopt',
	max_evaluations = 50,
 	# global_search_method = "genetic"
 	# global_search_method = "solver"
 	global_search_method = "sampling"
)
alg = rbfopt.RbfoptAlgorithm(settings, bb)
val, x, itercount, evalcount, fast_evalcount = alg.optimize()
x
##############################################################################
# Google OR Tools

n = 4
N = 10

model = cp_model.CpModel()
solver = cp_model.CpSolver()

x = [model.NewIntVar(0, N, f'x{i}') for i in range(n)]
printer = cp_model.VarArrayAndObjectiveSolutionPrinter(x)

x2 = [model.NewIntVar(0, N*N, f'x2_{i}') for i in range(n)]
for i in range(n): 
	model.AddMultiplicationEquality(x2[i], [x[i], x[i]])

model.Add(sum(x[i] for i in range(n)) == N)

status = solver.SearchForAllSolutions(model, printer)

model.Minimize(sum(x2[i] for i in range(n)))

# print(model)

status = solver.Solve(model)
status == cp_model.OPTIMAL or status == cp_model.FEASIBLE

y = [solver.Value(x[i]) for i in range(n)]
y

status = solver.SolveWithSolutionCallback(model, printer)

###############################################################################
# PAO : MILP: Pyomo-CBC ; NLP: scipy
###############################################################################

# число переменных
n = 4
# максимальный диапазон значений
N = 10

# диапазон индексов переменных решения
ix = range(n)
# начальное значение 
x = np.zeros(n)
x[0] = N
x_int = x[0:n//2]

# нелинейная выпуклая функция цели
# plt.plot(list(np.arange(0, 1, 0.01)), [i * pe.log(i + 1e-12) + (1-i) * pe.log(1-i + 1e-12)for i in np.arange(0,1,0.01)]); plt.show()
def obj(x):
 	return sum((x[i]/N) * np.log(x[i]/N + 1e-6) for i in ix)
# линейная аппроксимация функции цели
def get_linear_appr(x):
	return opt.approx_fprime(x, obj, epsilon = 1e-6)
# инициализация начального значения
def init_integer(model, i):
	return x[i]

###############################################
# MILP
# описание pyomo MILP модели
model = pe.ConcreteModel()

# целочисленные переменные решения
model.x = pe.Var(ix[:n-2], domain = pe.NonNegativeIntegers, bounds = (0, N), initialize = init_integer)
# непрерывные переменные решения
model.y = pe.Var(ix[n-2:n], domain = pe.NonNegativeReals, bounds = (0, N), initialize = 0)
# переменная решений - верхняя граница цели (максимум от линейных аппроксимаций)
model.mu = pe.Var(domain = pe.Reals, initialize = obj(x))

# ограничения MILP
model.cons = pe.Constraint(rule = 
   sum(model.x[i] for i in ix[:n-2]) + \
   sum(model.y[i] for i in ix[n-2:n]) == N
)
# пополняющийся после каждой MILP-итерации набор ограничений на аппроксимацию
model.cons_obj_lin = pe.ConstraintList()

# цель MILP
def mu_goal(model):
	return model.mu
model.obj = pe.Objective(rule = mu_goal, sense = pe.minimize)

###############################################
# NLP 
# только непрерывные переменные 

# границы
bounds = opt.Bounds([0] * (n // 2), [N] * (n // 2))

# цель NLP
def obj_cont(x):
	return obj(np.concatenate((x_int, x)))

# проверка на допустимость решения
def if_feasible(x_cont):
	x_cont = np.array(x_cont)
	c1 = np.prod(x_cont >= bounds.lb)
	c2 = np.prod(x_cont <= bounds.ub)
	c3 = np.prod(np.dot(lin_const.A, x_int) >= lin_const.lb)
	c4 = np.prod(np.dot(lin_const.A, x_int) <= lin_const.ub)
	c = c1 * c2 * c3 * c4
	return bool(c)

# запоминаем лучшее значение из NLP
def callback(xk, state):
	global best_x
	global best_val
	if state.fun < best_val and if_feasible(x_int):
		best_val = state.fun
		best_x = np.concatenate((x_int, state.x))
		print("NLP " + str(best_val))
	return False

###############################################
# Решение
# лучшее значение 
best_x = x
best_val = obj(x)
# Итерируем
prev_x = list()
for j in range(100):
	prev_x.append(copy.copy(x))	
	
	# MILP
	# линейная аппроксимация функции цели
	fx = obj(x)
	gradf = get_linear_appr(x)
	xgradf = np.dot(x, gradf)
	# добавляем новую аппроксимацию в ограничения
	model.cons_obj_lin.add(
		fx - \
		xgradf + \
		sum(model.x[i] * gradf[i] for i in ix[:n-2]) + \
		sum(model.y[i] * gradf[i] for i in ix[n-2:n]) <= model.mu
	)
	# новое решение MILP-задачи
	results = pe.SolverFactory('cbc').solve(model, tee = False) 
	x[:n-2] = np.array([model.x[i].value for i in ix[:n-2]])
	x[n-2:n] = np.array([model.y[i].value for i in ix[n-2:n]])
	print(x)
	# сохраняем лучшее решение
	if obj(x) < best_val:
		best_x = x
		best_val = obj(x)
		print("MILP " + str(best_val))
	# верхняя и нижняя граница
	print(obj(x), model.obj())
	
	# NLP
	# Фиксируем целочисленное решение
	x_int = x[:n-2]
	x_cont = x[2:n]
	# Переменные решения - только непрерывные
	# линейные ограничения NLP на непрерывные значения
	lin_const = opt.LinearConstraint(
		A = np.ones(n // 2),
		lb = N - sum(x_int[i] for i in range(2)),
		ub = N - sum(x_int[i] for i in range(2)),
		keep_feasible = True
	)
	# уточнение непрерывных переменных при помощи NLP
	res = opt.minimize(
		fun = obj_cont, 
		bounds = bounds,
		constraints = lin_const,
		x0 = x_cont,
		method = 'trust-constr',
		callback = callback,
		options = {'verbose': 0, "maxiter": 200}
	)
	print(np.concatenate((x_int, res.x)))
	
	# Функция цели - исходная нелинейная
	# Если решение лучше глобально-лучшего -- сохраняем его (улучшаем верхнюю границу)
	
	if (np.any([np.allclose(x, prev_x[i]) for i in range(len(prev_x))])):
		break
	if (abs(model.obj() - best_val) < 1e-3):
		break
	
print(best_x)



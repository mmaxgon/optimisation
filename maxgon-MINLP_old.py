'''
var x{j in 1..3} >= 0;
x[2], x[3] integer

# Objective function to be minimized.
minimize obj:
        1000 - x[1]^2 - 2*x[2]^2 - x[3]^2 - x[1]*x[2] - x[1]*x[3];

# Equality constraint.
s.t. c1: 8*x[1] + 14*x[2] + 7*x[3] - 56 = 0;

# Inequality constraint.
s.t. c2: x[1]^2 + x[2]^2 + x[3]^2 -25 >= 0;
'''
import copy
import numpy as np
import pyomo.environ as pyomo
import scipy.optimize as opt

###############################################################################

# число переменных
n = 3
# максимальный диапазон значений
N = 10

# диапазон индексов переменных решения
ix = range(n)
# начальное значение 
x = np.array([2]*3)
x_int = x[1:3]

# нелинейная выпуклая функция цели
def obj(x):
 	return -(1000 - x[0]**2 - 2*x[1]**2 - x[2]**2 - x[0]*x[1] - x[0]*x[2])
# нелинейные ограничения - неравенства
def non_lin_cons(x):
	return (x[0]**2 + x[1]**2 + x[2]**2 - 25)
# линейные ограничения - равенства
def lin_cons(x):
	return 8 * x[0] + 14 * x[1] + 7 * x[2] - 56

def if_feasible(x):
	x = np.array(x)
	c1 = (non_lin_cons(x) <= 0)
	c2 = (lin_cons(x) == 0)
	c3 = bool(np.prod(x >= 0))
	return (c1 and c2 and c3)

# градиент для линейной аппроксимации цели и ограничений
def get_linear_appr(obj, x):
	return opt.approx_fprime(x, obj, epsilon = 1e-6)

# инициализация начального значения
def init_integer(model, i):
	return x[i]

###############################################
# MILP
# описание pyomo MILP модели
model_milp = pyomo.ConcreteModel()

# Переменные решения
# целочисленные переменные решения
model_milp.x = pyomo.Var(ix[1:3], domain = pyomo.NonNegativeIntegers, bounds = (0, N), initialize = init_integer)
# непрерывные переменные решения
# model_milp.y = pyomo.Var(ix[:1], domain = pyomo.NonNegativeIntegers, bounds = (0, N), initialize = init_integer)
model_milp.y = pyomo.Var(ix[:1], domain = pyomo.NonNegativeReals, bounds = (0, N), initialize = init_integer)
# переменная решений - верхняя граница цели (максимум от линейных аппроксимаций)
model_milp.mu = pyomo.Var(domain = pyomo.Reals, bounds = (-1e6, 1e6), initialize = 0)

# Ограничения
# Ограничения на mu: должно быть лучше самого лучшего решения на данный момент
# model_milp.del_component(model_milp.mu_bound)
model_milp.mu_bound = pyomo.Constraint(rule = (model_milp.mu <= 1e6))
# model_milp.mu_bound.body.pprint()
# Линейные ограничения
model_milp.lin_cons = pyomo.ConstraintList()
model_milp.lin_cons.add(8 * model_milp.y[0] + 14 * model_milp.x[1] + 7 * model_milp.x[2] - 56 == 0)
# Нелинейные ограничения (пополняются для каждой итерации)
model_milp.non_lin_cons = pyomo.ConstraintList()
# ограничения на функцию цели (линейная аппроксимация после каждой итерации)
model_milp.obj_cons = pyomo.ConstraintList()

# цель MILP
def mu_goal(model):
	return model.mu
model_milp.obj = pyomo.Objective(rule = mu_goal, sense = pyomo.minimize)

###############################################
# Решение

x_best = None
goal_best = np.Inf
upper_bound = np.Inf
lower_bound = -np.Inf

# Цикл
while True:
	# MILP
	results = pyomo.SolverFactory('cbc').solve(model_milp, tee = True) 
	results
	
	if results.Solver()["Termination condition"] == pyomo.TerminationCondition.infeasible:
		print("MILP не нашёл решение")
		break
		
	# MILP-решение (даже недопустимое) даёт нижнюю границу
	lower_bound = pyomo.value(model_milp.obj)

	x = [model_milp.y[0](), model_milp.x[1](), model_milp.x[2]()]
	print(x)
	
	# если решение допустимо
	if non_lin_cons(x) <= 0:
		# добавляем новую аппроксимацию функции цели в ограничения
		fx = obj(x)
		gradf = get_linear_appr(obj, x)
		xgradf = np.dot(x, gradf)
		model_milp.obj_cons.add(
			fx - \
			xgradf + \
			model_milp.y[0] * gradf[0] + \
			sum(model_milp.x[i] * gradf[i] for i in [1, 2]) <= model_milp.mu
		)
		
		# NLP
		# уточняем непрерывные переменные
		# только непрерывные переменные 
		model_nlp = pyomo.ConcreteModel()
		model_nlp.y = pyomo.Var(domain = pyomo.NonNegativeReals, bounds = (0, N), initialize = model_milp.y[0]())
		model_nlp.lin_cons = pyomo.Constraint(rule = (8 * model_nlp.y + 14 * model_milp.x[1]() + 7 * model_milp.x[2]() - 56 == 0))
		model_nlp.non_lin_cons = pyomo.Constraint(rule = (model_nlp.y**2 + model_milp.x[1]()**2 + model_milp.x[2]()**2 - 25 <= 0))
		model_nlp.obj = pyomo.Objective(rule = (1000 - model_nlp.y**2 - 2*model_milp.x[1]()**2 - model_milp.x[2]()**2 - model_nlp.y*model_milp.x[1]() - model_nlp.y*model_milp.x[2]()), sense = pyomo.minimize)
		results = pyomo.SolverFactory('ipopt').solve(model_nlp, tee = True) 
		results
		assert results.Solver()["Termination condition"] == pyomo.TerminationCondition.optimal
		x[0] = model_nlp.y()
		fx = obj(x)
		
		# обновляем лучшее решение
		if fx < goal_best:
			x_best = copy.copy(x)
			goal_best = fx
			# верхняя граница - пока самое лучшее решение
			upper_bound = goal_best
			print(x_best, goal_best)			
	
		print(lower_bound, upper_bound)
		if (upper_bound - lower_bound < 1e-1):
			break
	
	# добавляем линеаризацию нелинейных ограничений
	gx = non_lin_cons(x)
	gradg = get_linear_appr(non_lin_cons, x)
	xgradg = np.dot(x, gradg)
	# добавляем новую аппроксимацию в ограничения
	model_milp.non_lin_cons.add(
		gx - \
		xgradg + \
		model_milp.y[0] * gradg[0] + \
		sum(model_milp.x[i] * gradg[i] for i in [1, 2]) <= 0
	)
	
print(x_best)


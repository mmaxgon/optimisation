import importlib

import maxgon_MINLP_POA as mg_minlp
importlib.reload(mg_minlp)
####################################################################################################
# MINLP with Polyhedral Outer Approximation
####################################################################################################
import copy
from time import time
import numpy as np
import scipy.optimize as opt

import pyomo.environ as pyomo
import ortools.sat.python.cp_model as ortools_cp_model
import docplex.mp.model as docplex_mp_model
from gekko import GEKKO
###############################################################################
'''
var x{j in 1..3} >= 0;
x[2], x[3] integer

# Objective function to be minimized.
maximize obj:
        1000 - x[1]^2 - 2*x[2]^2 - x[3]^2 - x[1]*x[2] - x[1]*x[3];

# Equality constraint.
s.t. c1: 8*x[1] + 14*x[2] + 7*x[3] - 56 = 0;

# Inequality constraint.
s.t. c2: x[1]^2 + x[2]^2 + x[3]^2 - 25 <= 0;
c3: x[2]^2 + x[3]^2 <= 12;	
'''
##############################################################################
# Задача как MILP Pyomo
##############################################################################
model_milp = pyomo.ConcreteModel()

# Переменные решения
# инициализация начального значения
x = np.array([2]*3)
def init_integer(model, i):
	return x[i]

# целочисленные переменные решения
model_milp.x = pyomo.Var([1, 2], domain = pyomo.NonNegativeIntegers, bounds = (0, 10), initialize = init_integer)
# непрерывные переменные решения
model_milp.y = pyomo.Var([0], domain = pyomo.NonNegativeReals, bounds = (0, 10), initialize = init_integer)
# Линейные ограничения
model_milp.lin_cons = pyomo.Constraint(expr = 8 * model_milp.y[0] + 14 * model_milp.x[1] + 7 * model_milp.x[2] - 56 == 0)

# model_milp.obj = pyomo.Objective(expr = 0, sense=pyomo.minimize)
# result = pyomo.SolverFactory("cbc").solve(model_milp)
# [model_milp.y[0](), model_milp.x[1](), model_milp.x[1]()]
# model_milp.del_component(model_milp.obj)

##############################################################################
# Задача как MINLP Pyomo
##############################################################################
model_minlp = pyomo.ConcreteModel()
# переменные решения
model_minlp.x = pyomo.Var([1, 2], domain = pyomo.NonNegativeIntegers, bounds = (0, 10), initialize = init_integer)
model_minlp.y = pyomo.Var([0], domain = pyomo.NonNegativeReals, bounds = (0, 10), initialize = init_integer)
# Линейные ограничения
model_minlp.lin_cons = pyomo.Constraint(expr = 8 * model_minlp.y[0] + 14 * model_minlp.x[1] + 7 * model_minlp.x[2] - 56 == 0)
# нелинейные ограничения
model_minlp.nlc1 = pyomo.Constraint(expr = model_minlp.y[0]**2 + model_minlp.x[1]**2 + model_minlp.x[2]**2 <= 25)
model_minlp.nlc2 = pyomo.Constraint(expr = model_minlp.x[1]**2 + model_minlp.x[2]**2 <= 12)
# нелинейная цель
model_minlp.obj = pyomo.Objective(expr = -(1000 - model_minlp.y[0]**2 - 2*model_minlp.x[1]**2 - model_minlp.x[2]**2 - model_minlp.y[0]*model_minlp.x[1] - model_minlp.y[0]*model_minlp.x[2]), sense=pyomo.minimize)

# pyomo.SolverFactory("couenne").solve(model_minlp)
# [model_minlp.y[0](), model_minlp.x[1](), model_minlp.x[1]()]

##############################################################################
# Задача как MILP CP-SAT
##############################################################################
model_milp_cpsat = ortools_cp_model.CpModel()

# Переменные решения (все целочисленные)
model_milp_cpsat_x = [model_milp_cpsat.NewIntVar(0, 10, "x[{}]".format(i)) for i in range(3)]

# Линейные ограничения
model_milp_cpsat_lin_cons = model_milp_cpsat.Add(8 * model_milp_cpsat_x[0] + 14 * model_milp_cpsat_x[1] + 7 * model_milp_cpsat_x[2] - 56 == 0)

# model_milp_cpsat_solver = ortools_cp_model.CpSolver()
# model_milp_cpsat_solver.parameters.max_time_in_seconds = 60.0
# status = model_milp_cpsat_solver.Solve(model_milp_cpsat)
# model_milp_cpsat_solver.StatusName()
# list(map(model_milp_cpsat_solver.Value, [model_milp_cpsat_x[0], model_milp_cpsat_x[1], model_milp_cpsat_x[1]]))

##############################################################################
# Задача как CPLEX MP
##############################################################################
model_cplex = docplex_mp_model.Model()

# Переменные решения (непрерывные)
model_cplex_y = model_cplex.var_list([0], lb=0, ub=10, vartype=model_cplex.continuous_vartype, name="y")
# Переменные решения (целочисленные)
model_cplex_x = model_cplex.var_list([1, 2], lb=0, ub=10, vartype=model_cplex.integer_vartype, name="x")

# Линейные ограничения
model_cplex_lin_cons = model_cplex.add(8 * model_cplex_y[0] + 14 * model_cplex_x[0] + 7 * model_cplex_x[1] - 56 == 0)

# DV_2_vec_cplex(model_cplex)
# [v.solution_value for v in DV_2_vec_cplex(model_cplex)]
# sol = model_cplex.solve()
# (sol["y_0"], sol["x_1"], sol["x_2"])

##############################################################################
# Задача как GEKKO MILP
##############################################################################
model_gekko = GEKKO(remote = False) # Initialize gekko
model_gekko.options.SOLVER = 1  # APOPT is an MINLP solver

# Переменные решения (непрерывные)
model_gekko.y = [model_gekko.Var(value=2, lb=0, ub=10, name="y_0")]
# Переменные решения (целочисленные)
model_gekko.x = [model_gekko.Var(value=2, lb=0, ub=10, integer=True, name="x_{}".format(i)) for i in [1,2]]

# Линейные ограничения
model_gekko.Equation(8 * model_gekko.y[0] + 14 * model_gekko.x[0] + 7 * model_gekko.x[1] - 56 == 0)

# model_gekko.Obj(-1e6)
# model_gekko.solve(disp=True)
# print(model_gekko.y[0].value)
# print(model_gekko.x[0].value)
# print(model_gekko.x[1].value)
# print(model_gekko.options.SOLVESTATUS)
# print('Objective: ' + str(model_gekko.options.objfcnval))

##############################################################################
# Задача как GEKKO MINLP
##############################################################################
model_gekko_minlp = GEKKO(remote = False) # Initialize gekko
model_gekko_minlp.options.SOLVER = 1  # APOPT is an MINLP solver

# Переменные решения (непрерывные)
model_gekko_minlp.y = [model_gekko_minlp.Var(value=2, lb=0, ub=10, name="y_0")]
# Переменные решения (целочисленные)
model_gekko_minlp.x = [model_gekko_minlp.Var(value=2, lb=0, ub=10, integer=True, name="x_{}".format(i)) for i in [1,2]]

# Линейные ограничения
model_gekko_minlp.Equation(8 * model_gekko_minlp.y[0] + 14 * model_gekko_minlp.x[0] + 7 * model_gekko_minlp.x[1] - 56 == 0)
# нелинейные ограничения
model_gekko_minlp.nlc1 = model_gekko_minlp.Equation(model_gekko_minlp.y[0]**2 + model_gekko_minlp.x[0]**2 + model_gekko_minlp.x[1]**2 <= 25)
model_gekko_minlp.nlc2 = model_gekko_minlp.Equation(model_gekko_minlp.x[0]**2 + model_gekko_minlp.x[1]**2 <= 12)
# нелинейная цель
model_gekko_minlp.Obj(-(1000 - model_gekko_minlp.y[0]**2 - 2*model_gekko_minlp.x[0]**2 - model_gekko_minlp.x[1]**2 - model_gekko_minlp.y[0]*model_gekko_minlp.x[0] - model_gekko_minlp.y[0]*model_gekko_minlp.x[1]))

# model_gekko_minlp.solve(disp=True)
# print(model_gekko_minlp.y[0].value)
# print(model_gekko_minlp.x[0].value)
# print(model_gekko_minlp.x[1].value)
# print(model_gekko_minlp.options.SOLVESTATUS)
# print('Objective: ' + str(model_gekko_minlp.options.objfcnval))

###############################################################################
# Функция, переводящая переменные решения pyomo в вектор
def DV_2_vec(model):
	x = [model.y[0], model.x[1], model.x[2]]
	return x

# Функция, переводящая переменные решения cp_sat в вектор
def DV_2_vec_cp_sat(model):
	x = [model.GetIntVarFromProtoIndex(0), model.GetIntVarFromProtoIndex(1), model.GetIntVarFromProtoIndex(2)]
	return x

# Функция, переводящая переменные решения cplex в вектор
def DV_2_vec_cplex(model):
	x = [model.get_var_by_index(0), model.get_var_by_index(1), model.get_var_by_index(2)]
	return x

def DV_2_vec_gekko(model):
	x = [model.y[0], model.x[0], model.x[1]]
	return x

# нелинейная выпуклая функция цели
def obj(x):
	return -(1000 - x[0]**2 - 2*x[1]**2 - x[2]**2 - x[0]*x[1] - x[0]*x[2])

# нелинейные ограничения - неравенства
def non_lin_cons(x):
	return [x[0]**2 + x[1]**2 + x[2]**2 - 25, x[1]**2 + x[2]**2 - 12]

def non_lin_cons_cp_sat(x):
	return [x[0]**2 + x[1]**2 + x[2]**2 - 25, x[1]**2 + x[2]**2 - 13]

###############################################################################
# scipy
###############################################################################
# Класс для уточнения непрерывных переменных решения при фиксации целочисленных
# Нужен когда непрерывных переменных много
class scipy_refiner_optimizer:
	# x - вектор переменных решения как непрерывных, так и целочисленных
	# мы фиксируем целочисленные и оптимизируем непрерывные
	def __init__(self, x):
		# фиксированные дискретные переменные
		self.x = [x[1], x[2]]
		# начальные значения непрерывных переменных
		self.y0 = self.get_cont_vars(x)
		# границы на непрерывные переменные
		self.bounds = opt.Bounds([0], [10])
		# линейные ограничения на непрерывные переменные (дискретные фиксированы и идут в правую часть)
		self.linear_constraint = opt.LinearConstraint([[8]], [56-14*self.x[0]-7*self.x[1]], [56-14*self.x[0]-7*self.x[1]])
		# нелинейные ограничения на непрерывные переменные (дискретные фиксированы и идут в правую часть)
		self.nonlinear_constraint = opt.NonlinearConstraint(lambda x: non_lin_cons(self.get_all_vars(x)), -np.inf, 0)

	# из вектора переменных решений получаем только непрерывные
	def get_cont_vars(self, x):
		return x[0]
	# непрерывные переменные решения конкатенируем с целыми - получаем полный вектор всех переменных решения
	def get_all_vars(self, x):
		return [x[0], self.x[0], self.x[1]]
	# целевая функция от непрерывных переменных с фикисированными целыми
	def get_goal(self, x):
		return obj(self.get_all_vars(x))
	# решение по непрерывным переменным
	def get_solution(self):
		res = opt.minimize(
			fun=self.get_goal,
			bounds=self.bounds,
			constraints=[self.linear_constraint, self.nonlinear_constraint],
			x0=self.y0,
			method="trust-constr",
			options = {'verbose': 0, "maxiter": 10}
		)
		res = {"x": self.get_all_vars(res.x), "obj": res.fun, "success": res.success}
		return res
# new_scipy_optimizer = scipy_refiner_optimizer([1.75, 2.0, 2.0])
# res = new_scipy_optimizer.get_solution()
# res

# Класс для проекции недопустимого решения на допустимую область с релаксацией целочисленности
# Нужен для уменьшения итераций по линейной аппроксимации ограничений
class scipy_projector_optimizer:
	# x - проецируемый недопустимый вектор
	def __init__(self):
		# границы на все переменные
		self.bounds = opt.Bounds([0, 0, 0], [10, 10, 10])
		# линейные ограничения
		self.linear_constraint = opt.LinearConstraint([[8, 14, 7]], [56], [56])
		# нелинейные ограничения
		self.nonlinear_constraint = opt.NonlinearConstraint(non_lin_cons, -np.inf, 0)

	# Проецируем x на допустимую область
	def get_solution(self, x):
		# начальный вектор переменных решения
		x0 = copy.copy(x)
		# целевая функция - расстояние до допустимого множества
		def get_goal(x):
			return sum((x[i] - x0[i]) ** 2 for i in range(len(x)))
		res = opt.minimize(
			fun=get_goal,
			bounds=self.bounds,
			constraints=[self.linear_constraint, self.nonlinear_constraint],
			x0=x0,
			method="trust-constr",
			options = {'verbose': 0, "maxiter": 100}
		)
		res = {"x": res.x, "obj": res.fun, "success": res.success}
		return res

scipy_projector_optimizer_obj = scipy_projector_optimizer()
# res = scipy_projector_optimizer_obj.get_solution([2, 2.0, 2.0])
# res

###############################################################################
importlib.reload(mg_minlp)
poa = mg_minlp.mmaxgon_MINLP_POA(
	eps=1e-6
)

###############################################################################
# GEKKO Нелинейная функция цели, есть нелинейные ограничения
###############################################################################
gekko_mip_model_wrapper = mg_minlp.gekko_MIP_model_wrapper(
	model_gekko=model_gekko,
	if_objective_defined=False
)

start_time = time()
res = poa.solve(
	MIP_model=gekko_mip_model_wrapper,
	non_lin_obj_fun=obj,
	non_lin_constr_fun=non_lin_cons,
	decision_vars_to_vector_fun=DV_2_vec_gekko,
	tolerance=1e-1,
	add_constr="ALL",
	NLP_refiner_class=None, #scipy_refiner_optimizer,
	NLP_projector_object=None #scipy_projector_optimizer_obj
)
print(time() - start_time)
print(res)

gekko_mip_model_wrapper = mg_minlp.gekko_MIP_model_wrapper(
	model_gekko=model_gekko_minlp,
	if_objective_defined=True
)

start_time = time()
res = poa.solve(
	MIP_model=gekko_mip_model_wrapper,
	non_lin_obj_fun=None,
	non_lin_constr_fun=None,
	decision_vars_to_vector_fun=DV_2_vec_gekko,
	tolerance=1e-1,
	add_constr="ALL",
	NLP_refiner_class=None, #scipy_refiner_optimizer,
	NLP_projector_object=None #scipy_projector_optimizer_obj
)
print(time() - start_time)
print(res)

###############################################################################
# CPLEX MP Нелинейная функция цели, есть нелинейные ограничения
###############################################################################
cplex_mip_model_wrapper = mg_minlp.cplex_MIP_model_wrapper(
	model_cplex=model_cplex
)

start_time = time()
res = poa.solve(
	MIP_model=cplex_mip_model_wrapper,
	non_lin_obj_fun=obj,
	non_lin_constr_fun=non_lin_cons,
	decision_vars_to_vector_fun=DV_2_vec_cplex,
	tolerance=1e-1,
	add_constr="ONE",
	NLP_refiner_class=None, #scipy_refiner_optimizer,
	NLP_projector_object=scipy_projector_optimizer_obj
)
print(time() - start_time)
print(res)

###############################################################################
# ortools cp_sat Нелинейная функция цели, есть нелинейные ограничения
###############################################################################
ortools_cp_sat_mip_model_wrapper = mg_minlp.ortools_cp_sat_MIP_model_wrapper(
	ortools_cp_model=ortools_cp_model,
	model_milp_cpsat=model_milp_cpsat,
	BIG_MULT=1e6
)

start_time = time()
res = poa.solve(
	MIP_model=ortools_cp_sat_mip_model_wrapper,
	non_lin_obj_fun=obj,
	non_lin_constr_fun=non_lin_cons_cp_sat,
	decision_vars_to_vector_fun=DV_2_vec_cp_sat,
	tolerance=1e-1,
	add_constr="ALL",
	NLP_refiner_class=None,
	NLP_projector_object=scipy_projector_optimizer_obj
)
print(time() - start_time)
print(res)

###############################################################################
# pyomo Нелинейная функция цели, есть нелинейные ограничения
###############################################################################
# с NLP
pyomo_mip_model_wrapper = mg_minlp.pyomo_MIP_model_wrapper(
	pyomo=pyomo,
	pyomo_MIP_model=model_milp,
	mip_solver_name="cbc" #"cplex"
)

start_time = time()
res1 = poa.solve(
	MIP_model=pyomo_mip_model_wrapper,
	non_lin_obj_fun=obj,
	non_lin_constr_fun=non_lin_cons,
	decision_vars_to_vector_fun=DV_2_vec,
	tolerance=1e-1,
	add_constr="ONE",
	NLP_refiner_class=None, #scipy_refiner_optimizer,
	NLP_projector_object=scipy_projector_optimizer_obj
)
print(time() - start_time)

pyomo_mip_model_wrapper = mg_minlp.pyomo_MIP_model_wrapper(
	pyomo=pyomo,
	pyomo_MIP_model=model_milp,
	mip_solver_name="cbc"
)
start_time = time()
res2 = poa.solve(
	MIP_model=pyomo_mip_model_wrapper,
	non_lin_obj_fun=obj,
	non_lin_constr_fun=non_lin_cons,
	decision_vars_to_vector_fun=DV_2_vec,
	tolerance=1e-1,
	add_constr="ALL",
	# NLP_refiner_class=scipy_refiner_optimizer,
	NLP_projector_object=scipy_projector_optimizer_obj
)
print(time() - start_time)

# без NLP
pyomo_mip_model_wrapper = mg_minlp.pyomo_MIP_model_wrapper(
	pyomo=pyomo,
	pyomo_MIP_model=model_milp,
	mip_solver_name="cbc"
)

start_time = time()
res3 = poa.solve(
	MIP_model=pyomo_mip_model_wrapper,
	non_lin_obj_fun=obj,
	non_lin_constr_fun=non_lin_cons,
	decision_vars_to_vector_fun=DV_2_vec,
	tolerance=1e-1,
	add_constr="ONE"
)
print(time() - start_time)

pyomo_mip_model_wrapper = mg_minlp.pyomo_MIP_model_wrapper(
	pyomo=pyomo,
	pyomo_MIP_model=model_milp,
	mip_solver_name="cbc"
)

start_time = time()
res4 = poa.solve(
	MIP_model=pyomo_mip_model_wrapper,
	non_lin_obj_fun=obj,
	non_lin_constr_fun=non_lin_cons,
	decision_vars_to_vector_fun=DV_2_vec,
	tolerance=1e-1,
	add_constr="ALL"
)
print(time() - start_time)

print(res1)
print(res2)
print(res3)
print(res4)

###########################
# Все ограничения линейные, функция цели нелинейная
###########################
pyomo_mip_model_wrapper = mg_minlp.pyomo_MIP_model_wrapper(
	pyomo=pyomo,
	pyomo_MIP_model=model_milp,
	mip_solver_name="cbc"
)
res5 = poa.solve(
	MIP_model=pyomo_mip_model_wrapper,
	non_lin_obj_fun=obj,
	non_lin_constr_fun=None,
	decision_vars_to_vector_fun=DV_2_vec,
	tolerance=1e-1,
	add_constr="ALL"
)
print(res5)

###########################
# Есть нелинейные ограничения, функция цели - линейная
###########################
model_milp.obj = pyomo.Objective(expr= model_milp.y[0] + 2*model_milp.x[1] + model_milp.x[2])
pyomo_mip_model_wrapper = mg_minlp.pyomo_MIP_model_wrapper(
	pyomo=pyomo,
	pyomo_MIP_model=model_milp,
	mip_solver_name="cbc"
)
res6 = poa.solve(
	MIP_model=pyomo_mip_model_wrapper,
	non_lin_obj_fun=None,
	non_lin_constr_fun=non_lin_cons,
	decision_vars_to_vector_fun=DV_2_vec,
	tolerance=1e-1,
	add_constr="ALL"
)
print(res6)

###########################
# Функция цели и все ограничения линейные
###########################
pyomo_mip_model_wrapper = mg_minlp.pyomo_MIP_model_wrapper(
	pyomo=pyomo,
	pyomo_MIP_model=model_milp,
	mip_solver_name="cbc"
)
res7 = poa.solve(
	MIP_model=pyomo_mip_model_wrapper,
	non_lin_obj_fun=None,
	non_lin_constr_fun=None,
	decision_vars_to_vector_fun=DV_2_vec,
	tolerance=1e-1,
	add_constr="ALL"
)
print(res7)

model_milp.del_component(model_milp.obj)

###############################################################################
# Используется базовый MINLP-солвер
# Нелинейная функция цели, есть нелинейные ограничения
###############################################################################
pyomo_mip_model_wrapper = mg_minlp.pyomo_MIP_model_wrapper(
	pyomo=pyomo,
	pyomo_MIP_model=model_minlp,
	mip_solver_name="couenne"
)

start_time = time()
res8 = poa.solve(
	MIP_model=pyomo_mip_model_wrapper,
	non_lin_obj_fun=None,
	non_lin_constr_fun=None,
	decision_vars_to_vector_fun=DV_2_vec,
	tolerance=1e-1,
	add_constr="ONE"
)
print(time() - start_time)

print(res8)
print(res3)

####################################################################
model_minlp.del_component(model_minlp.obj)
pyomo_mip_model_wrapper = mg_minlp.pyomo_MIP_model_wrapper(
	pyomo=pyomo,
	pyomo_MIP_model=model_minlp,
	mip_solver_name="couenne"
)

start_time = time()
res9 = poa.solve(
	MIP_model=pyomo_mip_model_wrapper,
	non_lin_obj_fun=obj,
	non_lin_constr_fun=None,
	decision_vars_to_vector_fun=DV_2_vec,
	tolerance=1e-1,
	add_constr="ONE"
)
print(time() - start_time)

print(res9)
print(res1)

model_minlp.obj = pyomo.Objective(expr = -(1000 - model_minlp.y[0]**2 - 2*model_minlp.x[1]**2 - model_minlp.x[2]**2 - model_minlp.y[0]*x[1] - model_minlp.y[0]*model_minlp.x[2]), sense=pyomo.minimize)


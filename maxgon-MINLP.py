import importlib

import maxgon_MINLP_POA as mg_minlp
importlib.reload(mg_minlp)
####################################################################################################
# MINLP with Polyhedral Outer Approximation
####################################################################################################
from collections import namedtuple
import copy
from time import time
import numpy as np
import scipy.optimize as opt

import pyomo.environ as pyomo
import ortools.sat.python.cp_model as ortools_cp_model
import docplex.mp.model as docplex_mp_model
import docplex.cp.model as docplex_cp_model
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

####################################################################################################
# Абстрактное описание задачи
####################################################################################################
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

####################################################################################################
# Получение нижней границы решения
####################################################################################################
# Нижняя граница как решение NLP-задачи
res_NLP = mg_minlp.get_NLP_lower_bound(opt_prob)
print(res_NLP)
nlp_lower_bound = res_NLP["obj"]

####################################################################################################
# Объект, уточняющий непрерывные компоненты решения при фиксации целочисленных
####################################################################################################
importlib.reload(mg_minlp)
scipy_refiner_optimizer_obj = mg_minlp.scipy_refiner_optimizer(opt_prob)
# refine = scipy_refiner_optimizer_obj.get_solution([1.75, 1.9999999980529146, 2.0000000038941708])
# print(refine)

####################################################################################################
# Объект, проецирующий недопустимое решение на допустимую область
####################################################################################################
scipy_projector_optimizer_obj = mg_minlp.scipy_projector_optimizer(opt_prob)
# project = scipy_projector_optimizer_obj.get_solution([5, 6, 7])
# print(project)

##############################################################################
# Задача как MILP Pyomo
##############################################################################
# инициализация начального значения
def init_integer(model, i):
	return opt_prob.dvars.x0[i]

def get_bounds(model, i):
	return (opt_prob.dvars.bounds.lb[i], opt_prob.dvars.bounds.ub[i])

model_milp = pyomo.ConcreteModel()

# целочисленные переменные решения
model_milp.x = pyomo.Var([1, 2], domain=pyomo.NonNegativeIntegers, bounds=get_bounds, initialize = init_integer)
# непрерывные переменные решения
model_milp.y = pyomo.Var([0], domain=pyomo.NonNegativeReals, bounds=get_bounds, initialize=init_integer)
# Линейные ограничения
model_milp.lin_cons = pyomo.Constraint(expr=
	opt_prob.linear_constraints.A[0][0] * model_milp.y[0] +
	sum(opt_prob.linear_constraints.A[0][i] * model_milp.x[i] for i in opt_prob.dvars.ix_int) == opt_prob.linear_constraints.bounds.ub[0])

# model_milp.obj = pyomo.Objective(expr = -model_milp.x[1], sense=pyomo.minimize)
# sf = pyomo.SolverFactory("cbc")
# sf.options["allowableGap"] = 1e-4
# sf.options["integerTolerance"] = 1e-4
# sf.options["seconds"] = 1e-1
# sf.options["outputFormat"] = 0
# sf.options["printingOptions"] = "all"
# start = time()
# result = sf.solve(model_milp, tee=False, warmstart=True)
# print(time() - start)
# [model_milp.y[0](), model_milp.x[1](), model_milp.x[1]()]
# model_milp.del_component(model_milp.obj)

##############################################################################
# Задача как MINLP Pyomo
##############################################################################
model_minlp = pyomo.ConcreteModel()
# переменные решения
model_minlp.x = pyomo.Var([1, 2], domain = pyomo.NonNegativeIntegers, bounds = (0, 10), initialize=lambda model, i: [3, 0][i-1])
model_minlp.y = pyomo.Var([0], domain = pyomo.NonNegativeReals, bounds = (0, 10), initialize=1.75)
# Линейные ограничения
model_minlp.lin_cons = pyomo.Constraint(expr = 8 * model_minlp.y[0] + 14 * model_minlp.x[1] + 7 * model_minlp.x[2] - 56 == 0)
# нелинейные ограничения
model_minlp.nlc1 = pyomo.Constraint(expr = model_minlp.y[0]**2 + model_minlp.x[1]**2 + model_minlp.x[2]**2 <= 25)
model_minlp.nlc2 = pyomo.Constraint(expr = model_minlp.x[1]**2 + model_minlp.x[2]**2 <= 12)
# нелинейная цель
model_minlp.obj = pyomo.Objective(expr = -(1000 - model_minlp.y[0]**2 - 2*model_minlp.x[1]**2 - model_minlp.x[2]**2 - model_minlp.y[0]*model_minlp.x[1] - model_minlp.y[0]*model_minlp.x[2]), sense=pyomo.minimize)

# start = time()
# pyomo.SolverFactory("cplex").solve(model_minlp, warmstart=True)
# pyomo.SolverFactory("couenne").solve(model_minlp)
# print(time() - start)
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
# Задача как CPLEX MP MILP
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
# Задача как CPLEX MP MINLP
##############################################################################
model_cplex_minlp = docplex_mp_model.Model()

# Переменные решения (непрерывные)
model_cplex_minlp_y = model_cplex_minlp.var_list([0], lb=0, ub=10, vartype=model_cplex_minlp.continuous_vartype, name="y")
# Переменные решения (целочисленные)
model_cplex_minlp_x = model_cplex_minlp.var_list([1, 2], lb=0, ub=10, vartype=model_cplex_minlp.integer_vartype, name="x")

# Линейные ограничения
model_cplex_minlp_lin_cons = model_cplex_minlp.add(8 * model_cplex_minlp_y[0] + 14 * model_cplex_minlp_x[0] + 7 * model_cplex_minlp_x[1] - 56 == 0)

# нелинейные ограничения
model_cplex_minlp_nlc1 = model_cplex_minlp.add(model_cplex_minlp_y[0]**2 + model_cplex_minlp_x[0]**2 + model_cplex_minlp_x[1]**2 <= 25)
model_cplex_minlp_nlc2 = model_cplex_minlp.add(model_cplex_minlp_x[0]**2 + model_cplex_minlp_x[1]**2 <= 12)
# нелинейная цель
model_cplex_minlp.minimize(-(1000 - model_cplex_minlp_y[0]**2 - 2*model_cplex_minlp_x[0]**2 - model_cplex_minlp_x[1]**2 - model_cplex_minlp_y[0]*model_cplex_minlp_x[0] - model_cplex_minlp_y[0]*model_cplex_minlp_x[1]))

# DV_2_vec_cplex(model_cplex)
# [v.solution_value for v in DV_2_vec_cplex(model_cplex)]
# sol = model_cplex_minlp.solve()
# (sol["y_0"], sol["x_1"], sol["x_2"])

##############################################################################
# Задача как CPLEX CP MILP
##############################################################################
model_cplex_cp = docplex_cp_model.CpoModel()

# Переменные решения (непрерывные)
model_cplex_cp.y = model_cplex_cp.integer_var_list(size=1, min=0, max=10, name="y")
# Переменные решения (целочисленные)
model_cplex_cp.x = model_cplex_cp.integer_var_list(size=2, min=0, max=10, name="x")

# Линейные ограничения
model_cplex_cp_lin_cons = model_cplex_cp.add(8 * model_cplex_cp.y[0] + 14 * model_cplex_cp.x[0] + 7 * model_cplex_cp.x[1] - 56 == 0)

# sol = model_cplex_cp.solve()
# DV_2_vec_gekko(model_cplex_cp)
# [sol[v.name] for v in DV_2_vec_gekko(model_cplex_cp)]
# (sol["y_0"], sol["x_0"], sol["x_1"])

##############################################################################
# Задача как CPLEX CP MINLP
##############################################################################
model_cplex_minlp_cp = docplex_cp_model.CpoModel()

# Переменные решения (непрерывные)
model_cplex_minlp_cp.y = model_cplex_minlp_cp.integer_var_list(size=1, min=0, max=10, name="y")
# Переменные решения (целочисленные)
model_cplex_minlp_cp.x = model_cplex_minlp_cp.integer_var_list(size=2, min=0, max=10, name="x")

# Линейные ограничения
model_cplex_minlp_cp_lin_cons = model_cplex_minlp_cp.add(8 * model_cplex_minlp_cp.y[0] + 14 * model_cplex_minlp_cp.x[0] + 7 * model_cplex_minlp_cp.x[1] - 56 == 0)

# нелинейные ограничения
model_cplex_minlp_cp_nlc1 = model_cplex_minlp_cp.add(model_cplex_minlp_cp.y[0]**2 + model_cplex_minlp_cp.x[0]**2 + model_cplex_minlp_cp.x[1]**2 <= 25)
model_cplex_minlp_cp_nlc2 = model_cplex_minlp_cp.add(model_cplex_minlp_cp.x[0]**2 + model_cplex_minlp_cp.x[1]**2 <= 13)
# нелинейная цель
model_cplex_minlp_cp.minimize(-(1000 - model_cplex_minlp_cp.y[0]**2 - 2*model_cplex_minlp_cp.x[0]**2 - model_cplex_minlp_cp.x[1]**2 - model_cplex_minlp_cp.y[0]*model_cplex_minlp_cp.x[0] - model_cplex_minlp_cp.y[0]*model_cplex_minlp_cp.x[1]))

# sol = model_cplex_minlp_cp.solve()
# DV_2_vec_gekko(model_cplex_minlp_cp)
# [sol[v.name] for v in DV_2_vec_gekko(model_cplex_minlp_cp)]
# (sol["y_0"], sol["x_0"], sol["x_1"])

##############################################################################
# Задача как GEKKO MILP - ПОХОЖЕ НЕ РАБОТАЕТ
##############################################################################
model_gekko = GEKKO(remote = False) # Initialize gekko
model_gekko.options.SOLVER = 1  # APOPT is an MINLP solver
model_gekko.options.MAX_ITER = 1000

model_gekko.solver_options = [
	'minlp_maximum_iterations 500', \
	# minlp iterations with integer solution
	'minlp_max_iter_with_int_sol 100', \
	# treat minlp as nlp
	'minlp_as_nlp 0', \
	# nlp sub-problem max iterations
	'nlp_maximum_iterations 50', \
	# 1 = depth first, 2 = breadth first
	'minlp_branch_method 1', \
	# maximum deviation from whole number
	'minlp_integer_tol 0.001', \
	# covergence tolerance
	'minlp_gap_tol 0.001'
]

# Переменные решения (непрерывные)
model_gekko.y = [model_gekko.Var(value=2, lb=0, ub=10, name="y_0")]
# Переменные решения (целочисленные)
model_gekko.x = [model_gekko.Var(value=2, lb=0, ub=10, integer=True, name="x_{}".format(i)) for i in [1,2]]

# Линейные ограничения
model_gekko.Equation(8 * model_gekko.y[0] + 14 * model_gekko.x[0] + 7 * model_gekko.x[1] - 56 == 0)

# model_gekko.Obj(-1e6)
# model_gekko.Equation("y_0 >= 2")
# model_gekko.solve(disp=True)
# print(model_gekko.y[0].value)
# print(model_gekko.x[0].value)
# print(model_gekko.x[1].value)
# print(model_gekko.options.SOLVESTATUS)
# print('Objective: ' + str(model_gekko.options.objfcnval))

##############################################################################
# Задача как GEKKO MINLP - ПОХОЖЕ НЕ РАБОТАЕТ
##############################################################################
model_gekko_minlp = GEKKO(remote = False) # Initialize gekko
model_gekko_minlp.options.SOLVER = 1  # APOPT is an MINLP solver
model_gekko_minlp.solver_options = [
	'minlp_maximum_iterations 500', \
	# minlp iterations with integer solution
	'minlp_max_iter_with_int_sol 10', \
	# treat minlp as nlp
	'minlp_as_nlp 0', \
	# nlp sub-problem max iterations
	'nlp_maximum_iterations 50', \
	# 1 = depth first, 2 = breadth first
	'minlp_branch_method 2', \
	# maximum deviation from whole number
	'minlp_integer_tol 0.05', \
	# covergence tolerance
	'minlp_gap_tol 0.01'
]

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

###############################################################################
# Решение
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
	non_lin_obj_fun=opt_prob.objective.fun,
	non_lin_constr_fun=opt_prob.nonlinear_constraints.fun,
	decision_vars_to_vector_fun=DV_2_vec_gekko,
	tolerance=1e-1,
	add_constr="ALL",
	# NLP_refiner_object=scipy_refiner_optimizer_obj,
	# NLP_projector_object=scipy_projector_optimizer_obj,
	lower_bound=nlp_lower_bound
	,custom_constraints_list=[gekko_mip_model_wrapper.get_mip_model().y[0] >= 1]
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
	NLP_refiner_object=None, #scipy_refiner_optimizer_obj,
	NLP_projector_object=None #scipy_projector_optimizer_obj
	,custom_constraints_list=["y_0 >= 2"]
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
	non_lin_obj_fun=opt_prob.objective.fun,
	non_lin_constr_fun=opt_prob.nonlinear_constraints.fun,
	decision_vars_to_vector_fun=DV_2_vec_cplex,
	tolerance=1e-1,
	add_constr="ALL",
	NLP_refiner_object=scipy_refiner_optimizer_obj,
	NLP_projector_object=scipy_projector_optimizer_obj,
	lower_bound=nlp_lower_bound
	,custom_constraints_list=[cplex_mip_model_wrapper.get_mip_model().get_var_by_index(0) >= 1]
)
print(time() - start_time)
print(res)

start_time = time()
res = poa.solve(
	MIP_model=cplex_mip_model_wrapper,
	non_lin_obj_fun=opt_prob.objective.fun,
	non_lin_constr_fun=opt_prob.nonlinear_constraints.fun,
	decision_vars_to_vector_fun=DV_2_vec_cplex,
	tolerance=1e-1,
	add_constr="ALL",
	NLP_refiner_object=scipy_refiner_optimizer_obj,
	NLP_projector_object=scipy_projector_optimizer_obj,
	lower_bound=nlp_lower_bound
	,custom_constraints_list=[cplex_mip_model_wrapper.get_mip_model().get_var_by_index(0) >= 2]
)
print(time() - start_time)
print(res)

start_time = time()
res = poa.solve(
	MIP_model=cplex_mip_model_wrapper,
	non_lin_obj_fun=opt_prob.objective.fun,
	non_lin_constr_fun=opt_prob.nonlinear_constraints.fun,
	decision_vars_to_vector_fun=DV_2_vec_cplex,
	tolerance=1e-1,
	add_constr="ALL",
	NLP_refiner_object=scipy_refiner_optimizer_obj,
	NLP_projector_object=scipy_projector_optimizer_obj,
	lower_bound=nlp_lower_bound
	,custom_constraints_list=[]
)
print(time() - start_time)
print(res)

cplex_mip_model_wrapper = mg_minlp.cplex_MIP_model_wrapper(
	model_cplex=model_cplex_minlp
)

start_time = time()
res = poa.solve(
	MIP_model=cplex_mip_model_wrapper,
	non_lin_obj_fun=None,
	non_lin_constr_fun=None,
	decision_vars_to_vector_fun=DV_2_vec_cplex,
	tolerance=1e-1,
	add_constr="ONE",
	NLP_refiner_object=None, #scipy_refiner_optimizer_obj,
	NLP_projector_object=scipy_projector_optimizer_obj
)
print(time() - start_time)
print(res)

###############################################################################
# CPLEX CP Нелинейная функция цели, есть нелинейные ограничения
###############################################################################

cplex_cp_mip_model_wrapper = mg_minlp.cplex_MIP_model_wrapper(
	model_cplex=model_cplex_cp
)
start_time = time()
res = poa.solve(
	MIP_model=cplex_cp_mip_model_wrapper,
	non_lin_obj_fun=opt_prob_cp.objective.fun,
	non_lin_constr_fun=opt_prob_cp.nonlinear_constraints.fun,
	decision_vars_to_vector_fun=DV_2_vec_gekko,
	tolerance=1e-1,
	add_constr="ALL",
	NLP_projector_object=scipy_projector_optimizer_obj,
	lower_bound=nlp_lower_bound
	,custom_constraints_list=[cplex_cp_mip_model_wrapper.get_mip_model().y[0] == 0]
)
print(time() - start_time)
print(res)

cplex_cp_mip_model_wrapper = mg_minlp.cplex_MIP_model_wrapper(
	model_cplex=model_cplex_cp
)
start_time = time()
res = poa.solve(
	MIP_model=cplex_cp_mip_model_wrapper,
	non_lin_obj_fun=opt_prob_cp.objective.fun,
	non_lin_constr_fun=opt_prob_cp.nonlinear_constraints.fun,
	decision_vars_to_vector_fun=DV_2_vec_gekko,
	tolerance=1e-1,
	add_constr="ALL",
	NLP_projector_object=scipy_projector_optimizer_obj,
	lower_bound=nlp_lower_bound
	,custom_constraints_list=[]
)
print(time() - start_time)
print(res)

cplex_cp_mip_model_wrapper = mg_minlp.cplex_MIP_model_wrapper(
	model_cplex=model_cplex_minlp_cp
)

start_time = time()
res = poa.solve(
	MIP_model=cplex_cp_mip_model_wrapper,
	non_lin_obj_fun=None,
	non_lin_constr_fun=None,
	decision_vars_to_vector_fun=DV_2_vec_gekko,
	tolerance=1e-1,
	add_constr="ALL",
	NLP_projector_object=None, #scipy_projector_optimizer_obj
	lower_bound=nlp_lower_bound
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
	non_lin_obj_fun=opt_prob_cp.objective.fun,
	non_lin_constr_fun=opt_prob_cp.nonlinear_constraints.fun,
	decision_vars_to_vector_fun=DV_2_vec_cp_sat,
	tolerance=1e-1,
	add_constr="ALL",
	NLP_projector_object=scipy_projector_optimizer_obj
	,custom_constraints_list=[ortools_cp_sat_mip_model_wrapper.get_mip_model().GetIntVarFromProtoIndex(0) == 0]
)
print(time() - start_time)
print(res)

ortools_cp_sat_mip_model_wrapper = mg_minlp.ortools_cp_sat_MIP_model_wrapper(
	ortools_cp_model=ortools_cp_model,
	model_milp_cpsat=model_milp_cpsat,
	BIG_MULT=1e6
)
start_time = time()
res = poa.solve(
	MIP_model=ortools_cp_sat_mip_model_wrapper,
	non_lin_obj_fun=opt_prob_cp.objective.fun,
	non_lin_constr_fun=opt_prob_cp.nonlinear_constraints.fun,
	decision_vars_to_vector_fun=DV_2_vec_cp_sat,
	tolerance=1e-1,
	add_constr="ALL",
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
	mip_solver_name="cbc",
	mip_solver_options={"allowableGap": 1e-1, "integerTolerance": 1e-1, "seconds": 1e-1}
)

start_time = time()
res1 = poa.solve(
	MIP_model=pyomo_mip_model_wrapper,
	non_lin_obj_fun=opt_prob.objective.fun,
	non_lin_constr_fun=opt_prob.nonlinear_constraints.fun,
	decision_vars_to_vector_fun=DV_2_vec,
	tolerance=1e-1,
	add_constr="ONE",
	NLP_refiner_object=scipy_refiner_optimizer_obj,
	NLP_projector_object=scipy_projector_optimizer_obj,
	lower_bound=nlp_lower_bound
	,custom_constraints_list=[pyomo_mip_model_wrapper.get_mip_model().y[0]>=1]
)
print(time() - start_time)
print(res1)

start_time = time()
res1 = poa.solve(
	MIP_model=pyomo_mip_model_wrapper,
	non_lin_obj_fun=opt_prob.objective.fun,
	non_lin_constr_fun=opt_prob.nonlinear_constraints.fun,
	decision_vars_to_vector_fun=DV_2_vec,
	tolerance=1e-1,
	add_constr="ONE",
	NLP_refiner_object=scipy_refiner_optimizer_obj,
	NLP_projector_object=scipy_projector_optimizer_obj,
	lower_bound=nlp_lower_bound
	,custom_constraints_list=[pyomo_mip_model_wrapper.get_mip_model().y[0]>=2]
)
print(time() - start_time)
print(res1)

start_time = time()
res1 = poa.solve(
	MIP_model=pyomo_mip_model_wrapper,
	non_lin_obj_fun=opt_prob.objective.fun,
	non_lin_constr_fun=opt_prob.nonlinear_constraints.fun,
	decision_vars_to_vector_fun=DV_2_vec,
	tolerance=1e-1,
	add_constr="ONE",
	NLP_refiner_object=scipy_refiner_optimizer_obj,
	NLP_projector_object=scipy_projector_optimizer_obj,
	lower_bound=nlp_lower_bound
	,custom_constraints_list=[pyomo_mip_model_wrapper.get_mip_model().y[0]>=1, pyomo_mip_model_wrapper.get_mip_model().y[0]<=2]
)
print(time() - start_time)
print(res1)

start_time = time()
res1 = poa.solve(
	MIP_model=pyomo_mip_model_wrapper,
	non_lin_obj_fun=opt_prob.objective.fun,
	non_lin_constr_fun=opt_prob.nonlinear_constraints.fun,
	decision_vars_to_vector_fun=DV_2_vec,
	tolerance=1e-1,
	add_constr="ONE",
	NLP_refiner_object=scipy_refiner_optimizer_obj,
	NLP_projector_object=scipy_projector_optimizer_obj,
	lower_bound=nlp_lower_bound
)
print(time() - start_time)
print(res1)

pyomo_mip_model_wrapper = mg_minlp.pyomo_MIP_model_wrapper(
	pyomo=pyomo,
	pyomo_MIP_model=model_milp,
	mip_solver_name="cbc"
)
start_time = time()
res2 = poa.solve(
	MIP_model=pyomo_mip_model_wrapper,
	non_lin_obj_fun=opt_prob.objective.fun,
	non_lin_constr_fun=opt_prob.nonlinear_constraints.fun,
	decision_vars_to_vector_fun=DV_2_vec,
	tolerance=1e-1,
	add_constr="ALL",
	NLP_refiner_object=scipy_refiner_optimizer_obj,
	NLP_projector_object=scipy_projector_optimizer_obj
)
print(time() - start_time)
print(res2)

# без NLP
pyomo_mip_model_wrapper = mg_minlp.pyomo_MIP_model_wrapper(
	pyomo=pyomo,
	pyomo_MIP_model=model_milp,
	mip_solver_name="cbc"
)

start_time = time()
res3 = poa.solve(
	MIP_model=pyomo_mip_model_wrapper,
	non_lin_obj_fun=opt_prob.objective.fun,
	non_lin_constr_fun=opt_prob.nonlinear_constraints.fun,
	decision_vars_to_vector_fun=DV_2_vec,
	tolerance=1e-1,
	add_constr="ONE"
)
print(time() - start_time)
print(res3)

pyomo_mip_model_wrapper = mg_minlp.pyomo_MIP_model_wrapper(
	pyomo=pyomo,
	pyomo_MIP_model=model_milp,
	mip_solver_name="cbc"
)

start_time = time()
res4 = poa.solve(
	MIP_model=pyomo_mip_model_wrapper,
	non_lin_obj_fun=opt_prob.objective.fun,
	non_lin_constr_fun=opt_prob.nonlinear_constraints.fun,
	decision_vars_to_vector_fun=DV_2_vec,
	tolerance=1e-1,
	add_constr="ALL"
)
print(time() - start_time)
print(res4)

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
	non_lin_obj_fun=opt_prob.objective.fun,
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
	non_lin_constr_fun=opt_prob.nonlinear_constraints.fun,
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
	# mip_solver_name="cplex",
	mip_solver_name="bonmin", #"couenne"
	mip_solver_options={"bonmin.algorithm":"B-OA"}
)

start_time = time()
res8 = poa.solve(
	MIP_model=pyomo_mip_model_wrapper,
	non_lin_obj_fun=None,
	non_lin_constr_fun=None,
	decision_vars_to_vector_fun=DV_2_vec,
	tolerance=1e-1,
	add_constr="ONE",
	NLP_refiner_object=scipy_refiner_optimizer_obj
)
print(time() - start_time)

print(res8)
print(res3)

####################################################################
model_minlp.del_component(model_minlp.obj)
pyomo_mip_model_wrapper = mg_minlp.pyomo_MIP_model_wrapper(
	pyomo=pyomo,
	pyomo_MIP_model=model_minlp,
	mip_solver_name="shot"
)

start_time = time()
res9 = poa.solve(
	MIP_model=pyomo_mip_model_wrapper,
	non_lin_obj_fun=opt_prob.objective.fun,
	non_lin_constr_fun=None,
	decision_vars_to_vector_fun=DV_2_vec,
	tolerance=1e-1,
	add_constr="ONE"
)
print(time() - start_time)

print(res9)
print(res1)

model_minlp.obj = pyomo.Objective(expr = -(1000 - model_minlp.y[0]**2 - 2*model_minlp.x[1]**2 - model_minlp.x[2]**2 - model_minlp.y[0]*x[1] - model_minlp.y[0]*model_minlp.x[2]), sense=pyomo.minimize)


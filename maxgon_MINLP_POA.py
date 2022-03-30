####################################################################################################
# MINLP with Polyhedral Outer Approximation
####################################################################################################
from collections import namedtuple
import copy
import numpy as np
import scipy.optimize as opt

import pyomo.environ as pyomo
# import ortools.sat.python.cp_model as ortools_cp_model
# import docplex.mp.model as docplex_mp_model
# from gekko import GEKKO

####################################################################################################
# Объекты абстрактного описания задачи
####################################################################################################

# Границы
bounds = namedtuple("bounds", ["lb", "ub"])
# Переменные решения
dvars = namedtuple("dvars", ["n", "ix_int", "ix_cont", "bounds", "x0"])
# Цель
objective = namedtuple("objective", ["n", "if_linear", "fun", "lin_coeffs"])
# Линейные ограничения
linear_constraints = namedtuple("linear_constraints", ["m", "A", "bounds"])
# Нелинейные ограничения
nonlinear_constraints = namedtuple("nonlinear_constraints", ["m", "fun", "bounds"])
# Описание оптимизационной задачи
optimization_problem = namedtuple("optimization_problem", ["dvars", "objective", "linear_constraints", "nonlinear_constraints"])

####################################################################################################
# Получение нижней границы решения
####################################################################################################

def get_relaxed_solution(opt_prob, custom_linear_constraints = None, custom_nonlinear_constraints = None):
	if opt_prob.objective.if_linear and (opt_prob.nonlinear_constraints is None):
		# линейная задача
		ix_ub = np.where(opt_prob.linear_constraints.bounds.lb < opt_prob.linear_constraints.bounds.ub)[0]
		if len(ix_ub) == 0:
			A_ub = None
			b_ub = None
		else:
			A_ub = np.array(opt_prob.linear_constraints.A)[ix_ub]
			b_ub = np.array(opt_prob.linear_constraints.bounds.ub)[ix_ub]
			
		ix_eq = np.where(opt_prob.linear_constraints.bounds.lb == opt_prob.linear_constraints.bounds.ub)[0]
		if len(ix_eq) == 0:
			A_eq = None
			b_eq = None
		else:
			A_eq = np.array(opt_prob.linear_constraints.A)[ix_eq]
			b_eq = np.array(opt_prob.linear_constraints.bounds.ub)[ix_eq]
		
		bounds = []
		for i in range(len(opt_prob.dvars.bounds.lb)):
			bounds.append((opt_prob.dvars.bounds.lb[i], opt_prob.dvars.bounds.ub[i]))

		c = opt_prob.objective.lin_coeffs
		
		res = opt.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method='simplex', callback=None, options=None, x0=opt_prob.dvars.x0)
		print(res)
		res = {"x": res.x, "obj": res.fun, "success": res.success, "constr_violation": np.max(np.abs(res.con))}
		return res
	
	# нелинейная задача
	constraints = [
		opt.LinearConstraint(
			opt_prob.linear_constraints.A,
			opt_prob.linear_constraints.bounds.lb,
			opt_prob.linear_constraints.bounds.ub
		),
		opt.NonlinearConstraint(
			opt_prob.nonlinear_constraints.fun,
			opt_prob.nonlinear_constraints.bounds.lb,
			opt_prob.nonlinear_constraints.bounds.ub
		)
	]
	if custom_linear_constraints != None:
		constraints.append(
			opt.LinearConstraint(
			custom_linear_constraints.A,
			custom_linear_constraints.bounds.lb,
			custom_linear_constraints.bounds.ub
		))
	if custom_nonlinear_constraints != None:
		constraints.append(opt.NonlinearConstraint(
			custom_nonlinear_constraints.fun,
			custom_nonlinear_constraints.bounds.lb,
			custom_nonlinear_constraints.bounds.ub
		))
	res = opt.minimize(
		fun=opt_prob.objective.fun,
		bounds=opt.Bounds(opt_prob.dvars.bounds.lb, opt_prob.dvars.bounds.ub),
		constraints=constraints,
		x0=opt_prob.dvars.x0,
		method="trust-constr",
		options={'verbose': 0, "maxiter": 200}
	)
	res = {"x": res.x, "obj": res.fun, "success": res.success, "constr_violation": res.constr_violation}
	return res

####################################################################################################
# Уточнение непрерывных компонент решения при фиксации целочисленных
####################################################################################################

# Класс для уточнения непрерывных переменных решения при фиксации целочисленных
# Нужен когда непрерывных переменных много: для уточнения верхней границы, для перевода решения в допустимую область
class scipy_refiner_optimizer:
	def __init__(self, opt_prob):
		# сохраняем начальное описание задачи
		self.__opt_prob = opt_prob

	# решение по непрерывным переменным
	def get_solution(self, x0):
		# фиксируем целочисленные переменные
		lb = np.array(self.__opt_prob.dvars.bounds.lb)
		ub = np.array(self.__opt_prob.dvars.bounds.ub)
		lb[self.__opt_prob.dvars.ix_int] = np.round(np.array(x0)[self.__opt_prob.dvars.ix_int])
		ub[self.__opt_prob.dvars.ix_int] = np.round(np.array(x0)[self.__opt_prob.dvars.ix_int])
		res = opt.minimize(
			fun=self.__opt_prob.objective.fun,
			bounds=opt.Bounds(lb, ub),
			constraints=[
				opt.LinearConstraint(
					self.__opt_prob.linear_constraints.A,
					self.__opt_prob.linear_constraints.bounds.lb,
					self.__opt_prob.linear_constraints.bounds.ub
				),
				opt.NonlinearConstraint(
					self.__opt_prob.nonlinear_constraints.fun,
					self.__opt_prob.nonlinear_constraints.bounds.lb,
					self.__opt_prob.nonlinear_constraints.bounds.ub
				)
			],
			x0=x0,
			method="trust-constr",
			options={'verbose': 0, "maxiter": 200}
		)
		res = {"x": res.x, "obj": res.fun, "success": res.success, "constr_violation": res.constr_violation}
		return res

####################################################################################################
# Проекция недопустимой точки на допустимую область (без учёта целочисленности)
####################################################################################################

# Класс для проекции недопустимого решения на допустимую область с релаксацией целочисленности
# Нужен для уменьшения итераций по линейной аппроксимации ограничений
class scipy_projector_optimizer:
	def __init__(self, opt_prob):
		# сохраняем начальное описание задачи
		self.__opt_prob = opt_prob

	# Проецируем x на допустимую область
	def get_solution(self, x0):
		# цель - расстояние до допустимой области
		def proj_obj(x):
			return sum((x[i] - x0[i]) ** 2 for i in range(self.__opt_prob.dvars.n))

		res = opt.minimize(
			fun=proj_obj,
			bounds=opt.Bounds(self.__opt_prob.dvars.bounds.lb, self.__opt_prob.dvars.bounds.ub),
			constraints=[
				opt.LinearConstraint(
					self.__opt_prob.linear_constraints.A,
					self.__opt_prob.linear_constraints.bounds.lb,
					self.__opt_prob.linear_constraints.bounds.ub
				),
				opt.NonlinearConstraint(
					self.__opt_prob.nonlinear_constraints.fun,
					self.__opt_prob.nonlinear_constraints.bounds.lb,
					self.__opt_prob.nonlinear_constraints.bounds.ub
				)
			],
			x0=self.__opt_prob.dvars.x0,
			method="trust-constr",
			options={'verbose': 0, "maxiter": 200}
		)
		res = {"x": res.x, "obj": res.fun, "success": res.success, "constr_violation": res.constr_violation}
		return res

####################################################################################################
# Попытка получить допустимое решение
####################################################################################################
"""
1. Берём NLP-решение
2. Находим переменную, самую близкую к целочисленной (из тех, что должны быть целочисленными)
3. Округляем её и фиксируем значение
4. Находим новое NLP-решение 
5. Если решение найдено: GOTO 2.
6. Если решение не найдено: округляем в другую сторону и решаем NLP. Если решение найдено: GOTO 2.
7. Если в обе стороны округления решение не найдено, идём по ветке вверх и меняем сторону округления там.
"""
def get_feasible_solution1(opt_prob, x_nlp):
	bounds_lb = np.array(opt_prob.dvars.bounds.lb)
	bounds_ub = np.array(opt_prob.dvars.bounds.ub)
	ix_int = copy.copy(opt_prob.dvars.ix_int)
	x_nlp = np.array(x_nlp)

	def build_int_solution_tree(x_nlp, ix_int, bounds_lb, bounds_ub):
		# значения переменных решения, которые должны быть целочисленными
		x_int = x_nlp[ix_int]
		# Отклонение значений от целочисленных
		x_dist_int = np.array([[np.ceil(x) - x, x - np.floor(x)] for x in x_int])
		# индекс мимнального значения
		ix_min = np.argmin(x_dist_int)
		# индекс соответствующей переменной
		ix_int_var = ix_min // 2
		# индекс округления (0 ceil, 1 floor)
		ix_int_var_dir = ix_min % 2
		# округляем
		x_val = x_int[ix_int_var]
		round_val = np.ceil(x_val) if ix_int_var_dir == 0 else np.floor(x_val)
		# формируем новые границы
		new_bounds_lb = np.array(bounds_lb)
		new_bounds_lb[ix_int[ix_int_var]] = round_val
		new_bounds_ub = np.array(bounds_ub)
		new_bounds_ub[ix_int[ix_int_var]] = round_val
		# получаем новую NLP задачу с фиксированными значениями round_val для целочисленной переменной ix_int_var
		new_opt_prob = optimization_problem(
			dvars(
				opt_prob.dvars.n,
				opt_prob.dvars.ix_int,
				opt_prob.dvars.ix_cont,
				bounds(new_bounds_lb, new_bounds_ub),
				x_nlp
			),
			opt_prob.objective,
			opt_prob.linear_constraints,
			opt_prob.nonlinear_constraints
		)
		# Получаем новое NLP-решение
		res = get_relaxed_solution(new_opt_prob)
		# Если допустимое решение нашли
		if res["success"] and (res["constr_violation"] <= 1e-6):
			print(res)
			# если не осталось больше целочисленных компонент, которые мы ещё не зафиксировали - задача решена
			if len(ix_int) <= 1:
				return res
			# если есть ещё целочисленные нефиксированные компоненты - формируем новую рекурсивную задачу
			new_x_nlp = res["x"]
			# удаляем из списка индексов целочисленных переменных фиксированную на этом шаге
			new_ix_int = np.delete(ix_int, ix_int_var)
			# решаем задачу дальше
			res = build_int_solution_tree(new_x_nlp, new_ix_int, new_bounds_lb, new_bounds_ub)
			# если внизу решение нашли - возвращаем его
			if not (res is None):
				return res
			# Если внизу решение не было найдено - меняем сторону округления на том шаге и снова идём вниз
			round_val = np.ceil(x_val) if ix_int_var_dir == 1 else np.floor(x_val)
			# формируем новые границы
			new_bounds_lb = np.array(bounds_lb)
			new_bounds_lb[ix_int[ix_int_var]] = round_val
			new_bounds_ub = np.array(bounds_ub)
			new_bounds_ub[ix_int[ix_int_var]] = round_val
			# получаем новую NLP задачу с фиксированными значениями round_val для целочисленной переменной ix_int_var
			new_opt_prob = optimization_problem(
				dvars(
					opt_prob.dvars.n,
					opt_prob.dvars.ix_int,
					opt_prob.dvars.ix_cont,
					bounds(new_bounds_lb, new_bounds_ub),
					x_nlp
				),
				opt_prob.objective,
				opt_prob.linear_constraints,
				opt_prob.nonlinear_constraints
			)
			# Получаем новое NLP-решение
			res = get_relaxed_solution(new_opt_prob)
			# если допустимое решение не найдено - идём наверх с пустым
			if not (res["success"] and (res["constr_violation"] <= 1e-6)):
				return None
			# если допустимое решение найдено
			print(res)
			# если не осталось больше целочисленных компонент, которые мы ещё не зафиксировали - задача решена
			if len(ix_int) <= 1:
				return res
			# если есть ещё целочисленные нефиксированные компоненты - формируем новую рекурсивную задачу
			new_x_nlp = res["x"]
			# удаляем из списка индексов целочисленных переменных фиксированную на этом шаге
			new_ix_int = np.delete(ix_int, ix_int_var)
			# решаем задачу дальше
			res = build_int_solution_tree(new_x_nlp, new_ix_int, new_bounds_lb, new_bounds_ub)
			# при любом решении внизу - возвращаем его наверх, так как все варианты уже испробовали
			return res
		else:
			# Если решение не было найдено - меняем сторону округления на этом шаге и снова решаем
			round_val = np.ceil(x_val) if ix_int_var_dir == 1 else np.floor(x_val)
			# формируем новые границы
			new_bounds_lb = np.array(bounds_lb)
			new_bounds_lb[ix_int[ix_int_var]] = round_val
			new_bounds_ub = np.array(bounds_ub)
			new_bounds_ub[ix_int[ix_int_var]] = round_val
			# получаем новую NLP задачу с фиксированными значениями round_val для целочисленной переменной ix_int_var
			new_opt_prob = optimization_problem(
				dvars(
					opt_prob.dvars.n,
					opt_prob.dvars.ix_int,
					opt_prob.dvars.ix_cont,
					bounds(new_bounds_lb, new_bounds_ub),
					x_nlp
				),
				opt_prob.objective,
				opt_prob.linear_constraints,
				opt_prob.nonlinear_constraints
			)
			# Получаем новое NLP-решение
			res = get_relaxed_solution(new_opt_prob)
			# если допустимое решение не найдено - идём наверх с пустым
			if not (res["success"] and (res["constr_violation"] <= 1e-6)):
				return None
			# если допустимое решение найдено
			print(res)
			# если не осталось больше целочисленных компонент, которые мы ещё не зафиксировали - задача решена
			if len(ix_int) <= 1:
				return res
			# если есть ещё целочисленные нефиксированные компоненты - формируем новую рекурсивную задачу
			new_x_nlp = res["x"]
			# удаляем из списка индексов целочисленных переменных фиксированную на этом шаге
			new_ix_int = np.delete(ix_int, ix_int_var)
			# решаем задачу дальше
			res = build_int_solution_tree(new_x_nlp, new_ix_int, new_bounds_lb, new_bounds_ub)
			# при любом решении внизу - возвращаем его наверх, так как все варианты уже испробовали
			return res
		
	return build_int_solution_tree(x_nlp, ix_int, bounds_lb, bounds_ub)

"""
Последовательное приближение друг к другу NLP и MILP решений.
1. Берём оптимальное NLP-решение
2. Находим самое близкое к NLP-решению целочисленное решение MILP только с линейными ограничениями. Если оно допустимо - возвращаем его.
Если нет - добавляем к задаче линеаризованные ограничения в виде касательных.
3. Находим новое NLP-решение как самую близкую точку к решению MILP из 2.
4. GOTO 2.
"""
def get_feasible_solution2(opt_prob, x_nlp):
	# NLP-описание задачи (проектор)
	NLP_projector = scipy_projector_optimizer(opt_prob)

	# MILP-описание задачи
	model_milp = pyomo.ConcreteModel()
	# целочисленные переменные решения
	model_milp.x_int = pyomo.Var(
		opt_prob.dvars.ix_int,
		domain=pyomo.Integers,
		bounds=lambda model, i: (opt_prob.dvars.bounds.lb[i], opt_prob.dvars.bounds.ub[i]),
		initialize=lambda model, i: opt_prob.dvars.x0[i]
	)
	# непрерывные переменные решения
	model_milp.x_cont = pyomo.Var(
		[i for i in range(opt_prob.dvars.n) if i not in opt_prob.dvars.ix_int],
		domain=pyomo.Reals,
		bounds=lambda model, i: (opt_prob.dvars.bounds.lb[i], opt_prob.dvars.bounds.ub[i]),
		initialize=lambda model, i: opt_prob.dvars.x0[i]
	)
	ix_cont = [i for i in model_milp.x_cont_index]
	ix_int = [i for i in model_milp.x_int_index]
	# все переменные решения
	x_var = []
	for i in range(opt_prob.dvars.n):
		if i in ix_cont:
			x_var.append(model_milp.x_cont[i])
		elif i in ix_int:
			x_var.append(model_milp.x_int[i])
		else:
			raise ValueError("Недопустимый индекс {0}!".format(i))
	# линейные ограничения
	model_milp.lin_cons = pyomo.ConstraintList()
	if model_milp.lin_cons != None:
		for j in range(opt_prob.linear_constraints.m):
			expr = np.matmul(opt_prob.linear_constraints.A[j], x_var)
			expr1 = expr <= opt_prob.linear_constraints.bounds.ub[j]
			expr2 = expr >= opt_prob.linear_constraints.bounds.lb[j]
			model_milp.lin_cons.add(expr1)
			model_milp.lin_cons.add(expr2)
	# вспомогательные ограничения при линеаризации
	model_milp.non_lin_cons = pyomo.ConstraintList()
	# ограничения цели
	model_milp.mu = pyomo.Var(domain=pyomo.NonNegativeReals)
	model_milp.obj = pyomo.Objective(expr=model_milp.mu, sense=pyomo.minimize)
	model_milp.obj_cons = pyomo.ConstraintList()
	prev_dist = np.Inf

	while True:
		# MILP итерация решения
		model_milp.obj_cons.clear()
		# Функция цели - L1-расстояние до NLP-решения
		for i in range(opt_prob.dvars.n):
			model_milp.obj_cons.add(x_var[i] - x_nlp[i] <= model_milp.mu)
			model_milp.obj_cons.add(x_var[i] - x_nlp[i] >= -model_milp.mu)
		# решаем: находим самое близкое MILP-решение к NLP-решению
		sf = pyomo.SolverFactory("cbc")
		result = sf.solve(model_milp, tee=False, warmstart=True)
		if result.Solver()["Termination condition"] == pyomo.TerminationCondition.infeasible:
			raise ValueError("Не найдено MILP-решение!")
		x_milp = [pyomo.value(x) for x in x_var]
		print("MILP: " + str(x_milp))
		if (opt_prob.nonlinear_constraints == None) or (
			np.all(np.array(opt_prob.nonlinear_constraints.fun(x_milp)) <= np.array(opt_prob.nonlinear_constraints.bounds.ub)) and \
			np.all(np.array(opt_prob.nonlinear_constraints.fun(x_milp)) >= np.array(opt_prob.nonlinear_constraints.bounds.lb))
		):
			return x_milp
		dist = np.sqrt(sum((x_nlp[i] - x_milp[i])**2 for i in range(opt_prob.dvars.n)))
		print(dist)
		if dist >= prev_dist:
			raise ValueError("Расстояние между MILP и NLP решениями не уменьшается!")
		prev_dist = dist
		# ДАЛЕЕ ПРЕДПОЛАГАЕМ, ЧТО НЕЛИНЕЙНЫЕ ОГРАНИЧЕНИЯ ВЫГЛЯДЯТ КАК g(x) <= 0, Т.Е., ВЕРХНЯЯ ГРАНИЦА 0, А НИЖНЕЙ НЕТ
		ix_violated = np.where(np.array(opt_prob.nonlinear_constraints.fun(x_milp)) > 0)[0]
		# добавляем линеаризованные ограничения
		gradg = opt.slsqp.approx_jacobian(x_milp, lambda x: np.array(opt_prob.nonlinear_constraints.fun(x))[ix_violated], epsilon=1e-9)
		gx = np.array(opt_prob.nonlinear_constraints.fun(x_milp))[ix_violated]
		for i in range(len(ix_violated)):
			expr = gx + sum(gradg[i][j] * (x_var[j] - x_milp[j]) for j in range(opt_prob.dvars.n)) <= 0
			model_milp.non_lin_cons.add(expr)

		# итерация NLP-приближения к MILP-решению
		nlp_projector = NLP_projector.get_solution(x_milp)
		if (not nlp_projector["success"]) or (nlp_projector["constr_violation"] >= 1e-6):
			raise ValueError("Не найдено NLP-решение!!")
		x_nlp = nlp_projector["x"]
		print("NLP: " + str(x_nlp))

####################################################################################################
# Решение простой MINLP задачи с помощью pyomo+cbc на основании описания задачи в виде optimization_problem
####################################################################################################
# случайные точки внутри диапазона
def generate_x(opt_prob):
	x = []
	for i in range(opt_prob.dvars.n):
		lb = opt_prob.dvars.bounds.lb[i]
		ub = opt_prob.dvars.bounds.ub[i]
		res = np.random.randint(lb, ub) if i in opt_prob.dvars.ix_int else lb + (ub - lb) * np.random.random_sample()
		x.append(res)
	return x
	
def get_minlp_solution(
	opt_prob,                       # описание задачи
	obj_tolerance=1e-6,             # разница между upper_bound и lower_bound
	if_nlp_lower_bound=False,       # нужно ли рассчитывать нижнюю границу NLP-задачи в начале
	if_refine=False,                # нужно ли после каждого MILP-решения фиксировать целочисленные переменные и уточнять непрерывные
	if_project=False,               # нужно ли проецировать недопустимое решение на допустимое множество и строить касательные к нелинейным ограничениям в точке проекции
	random_points_count=0           # сколько случайных точек сгенерировать вместе с касательными к функции цели и нелинейным ограничениям в них до начала решения MILP-задачи
):
	# добавление линейных ограничений в качестве касательных к нелинейной функции цели в точке x
	def add_obj_constraints(x, dvar_x, fx=None):
		if fx is None:
			fx = opt_prob.objective.fun(x)
		gradf = opt.approx_fprime(x, opt_prob.objective.fun, epsilon=1e-9)
		expr = fx + sum(gradf[j] * (dvar_x[j] - x[j]) for j in range(opt_prob.dvars.n)) <= model_milp.mu
		# print(expr)
		model_milp.obj_cons.add(expr)
	
	# добавление линейных ограничений в качестве касательных к нелинейным ограничениям в точке x
	def add_nonlin_constraints(x, dvar_x, ix=None, gx=None):
		if ix is None:
			ix = np.array(range(opt_prob.nonlinear_constraints.m))
		else:
			ix = np.array(ix)
			
		if gx is None:
			gx = np.array(opt_prob.nonlinear_constraints.fun(x))[ix]
		else:
			gx = np.array(gx)[ix]
		
		gradg = opt.slsqp.approx_jacobian(
			x,
			lambda y: np.array(opt_prob.nonlinear_constraints.fun(y))[ix],
			epsilon=1e-9
		)
		ub = np.array(opt_prob.nonlinear_constraints.bounds.ub)[ix]
		for i in range(len(ix)):
			expr = gx[i] + sum(gradg[i][j] * (x_var[j] - x_milp[j]) for j in range(opt_prob.dvars.n)) <= ub[i]
			# print(expr)
			if (str(expr) != "True"):
				model_milp.non_lin_cons.add(expr)
	
	# Рассчитываем решение NLP-задачи для получения нижней границы
	if if_nlp_lower_bound:
		res_NLP = get_relaxed_solution(opt_prob)
		if not (res_NLP["success"] and res_NLP["constr_violation"] <= 1e-6):
			raise ValueError("Нет NLP-решения!")
		nlp_lower_bound = res_NLP["obj"]
	# Объект для уточнения непрерывных переменных решения при фиксации целочисленных
	if if_refine:
		scipy_refiner_optimizer_obj = scipy_refiner_optimizer(opt_prob)
	# Объект проекции недопустимого решения на допустимое множество
	if if_project:
		scipy_projector_optimizer_obj = scipy_projector_optimizer(opt_prob)
		
	# MILP-описание задачи
	model_milp = pyomo.ConcreteModel()
	# целочисленные переменные решения
	model_milp.x_int = pyomo.Var(
		opt_prob.dvars.ix_int,
		domain=pyomo.Integers,
		bounds=lambda model, i: (opt_prob.dvars.bounds.lb[i], opt_prob.dvars.bounds.ub[i]),
		initialize=lambda model, i: opt_prob.dvars.x0[i]
	)
	# непрерывные переменные решения
	model_milp.x_cont = pyomo.Var(
		[i for i in range(opt_prob.dvars.n) if i not in opt_prob.dvars.ix_int],
		domain=pyomo.Reals,
		bounds=lambda model, i: (opt_prob.dvars.bounds.lb[i], opt_prob.dvars.bounds.ub[i]),
		initialize=lambda model, i: opt_prob.dvars.x0[i]
	)
	# индекси для непрерывных и целочисленных переменных
	ix_cont = [i for i in model_milp.x_cont_index]
	ix_int = [i for i in model_milp.x_int_index]
	# все переменные решения
	x_var = []
	for i in range(opt_prob.dvars.n):
		if i in ix_cont:
			x_var.append(model_milp.x_cont[i])
		elif i in ix_int:
			x_var.append(model_milp.x_int[i])
		else:
			raise ValueError("Недопустимый индекс {0}!".format(i))
	# линейные ограничения
	model_milp.lin_cons = pyomo.ConstraintList()
	if model_milp.lin_cons != None:
		for j in range(opt_prob.linear_constraints.m):
			expr = np.matmul(opt_prob.linear_constraints.A[j], x_var)
			if opt_prob.linear_constraints.bounds.ub[j] < np.Inf:
				expr1 = expr <= opt_prob.linear_constraints.bounds.ub[j]
				model_milp.lin_cons.add(expr1)
			if opt_prob.linear_constraints.bounds.lb[j] > -np.Inf:
				expr2 = expr >= opt_prob.linear_constraints.bounds.lb[j]
				model_milp.lin_cons.add(expr2)
	# ограничения при линеаризации нелинейных ограничений
	model_milp.non_lin_cons = pyomo.ConstraintList()
	# ограничения при линеаризации нелинейной функции цели
	model_milp.obj_cons = pyomo.ConstraintList()
	# вспомогательная функция цели
	model_milp.mu = pyomo.Var(domain=pyomo.Reals)
	if opt_prob.objective.if_linear:
		model_milp.obj_cons.add(expr=opt_prob.objective.fun(x_var) <= model_milp.mu)
	else:
		model_milp.temp_mu = pyomo.Constraint(expr=model_milp.mu >= -1e9)
	# минимизируем вспомогательную переменную mu
	model_milp.obj = pyomo.Objective(expr=model_milp.mu, sense=pyomo.minimize)
	
	# начальные значения
	# lower_bound = nlp_lower_bound if if_nlp_lower_bound else -np.Inf
	prev_obj = -np.inf
	obj = np.Inf
	if if_nlp_lower_bound:
		lower_bound = nlp_lower_bound
	else:
		lower_bound = -np.Inf
	upper_bound = np.Inf
	best_sol = None
	if_first_step = True
	sf = pyomo.SolverFactory("cbc")
	step_num = 0
	
	while True:
		step_num += 1

		# Сначала генерим случайные точки и строим в них касательные к нелинейным объектам,
		# чтобы лучше аппроксимировать нелинейные ограничения и функцию цели до начала решения MILP-задачи
		if step_num <= random_points_count:
			x_milp = generate_x(opt_prob)
			print("Random: " + str(x_milp))
			# добавляем касательную к функции цели
			if not opt_prob.objective.if_linear:
				add_obj_constraints(x=x_milp, dvar_x=x_var)
			# добавляем касательные к нелинейным ограничениям
			if not (opt_prob.nonlinear_constraints is None):
				add_nonlin_constraints(x=x_milp, dvar_x=x_var)
			continue

		# MILP итерация решения
		result = sf.solve(model_milp, tee=False, warmstart=True)
		if result.Solver()["Termination condition"] == pyomo.TerminationCondition.infeasible:
			raise ValueError("Не найдено MILP-решение!")
		# После первого шага удаляем временное ограничение на функцию цели снизу
		if if_first_step and not opt_prob.objective.if_linear:
			if_first_step = False
			model_milp.del_component(model_milp.temp_mu)
		# значения переменных решения
		x_milp = list(map(pyomo.value, x_var))
		print("MILP: " + str(x_milp))
		# значение целевой функции вспомогательной задачи
		obj = model_milp.mu.value
		if obj < prev_obj:
			raise ValueError("Значение целевой функции вспомогательной задачи не может уменьшаться!")
		prev_obj = obj
		lower_bound = max(lower_bound, obj)

		# фиксируем целочисленные переменные, оптимизируем по непрерывным
		if if_refine:
			refine = scipy_refiner_optimizer_obj.get_solution(x_milp)
			if refine["success"] and refine["constr_violation"] < 1e-6:
				x_refine = refine["x"]
				print("Refined: {0}".format(x_refine))
				x_milp = x_refine
		
		# значение целевой функции исходной задачи (совпадает с вспомогательной в случае линейности цели)
		fx = opt_prob.objective.fun(x_milp)
		
		# добавляем линейное ограничение на функцию цели в виде касательной в точке x_milp
		if not opt_prob.objective.if_linear:
			add_obj_constraints(x=x_milp, dvar_x=x_var, fx=fx)
			
		if not (opt_prob.nonlinear_constraints is None):
			# значение нелинейных ограничений в x_milp
			gx = np.array(opt_prob.nonlinear_constraints.fun(x_milp))
			# индексы нарушенных нелинейных ограничений
			# ДАЛЕЕ ПРЕДПОЛАГАЕМ, ЧТО НЕЛИНЕЙНЫЕ ОГРАНИЧЕНИЯ ВЫГЛЯДЯТ КАК g(x) <= ub, Т.Е., ВЕРХНЯЯ ГРАНИЦА ub, А НИЖНЕЙ НЕТ
			ix_violated = np.where(gx > opt_prob.nonlinear_constraints.bounds.ub)[0]
		else:
			ix_violated = []
		
		# Проверка на допустимость нелинейных ограничений
		if len(ix_violated) == 0:
			print("feasible")
			if fx < upper_bound:
				upper_bound = fx
				best_sol = copy.copy(x_milp)
				
			if (lower_bound > upper_bound):
				raise ValueError("lower_bound > upper_bound!")
			if (upper_bound - lower_bound < obj_tolerance):
				res = {"obj": upper_bound, "x": best_sol, "step_num": step_num-random_points_count, "nonlin_constr_num": len(model_milp.non_lin_cons), "objective_constr_num": len(model_milp.obj_cons)}
				print("lower_bound: {0}, upper_bound: {1}".format(lower_bound, upper_bound))
				return res
		else:
			# проецируем решение вспомогательной задачи на допустимую область
			if if_project:
				project = scipy_projector_optimizer_obj.get_solution(x_milp)
				if project["success"] and project["constr_violation"] < 1e-6:
					x_project = project["x"]
					print("Projected: {0}".format(x_project))
					x_milp = x_project
					# пересчитываем значения нелинейных ограничений в x_milp
					gx = np.array(opt_prob.nonlinear_constraints.fun(x_milp))
			# добавляем линейные ограничения в виде касательных к нарушенным нелинейным ограничениям в точке x_milp
			add_nonlin_constraints(x=x_milp, dvar_x=x_var, ix=ix_violated, gx=gx)

		print("lower_bound: {0}, upper_bound: {1}".format(lower_bound, upper_bound))

####################################################################################################
# Решение сложных MINLP задач с помощью описания на одном из фреймворков PYOMO, CP_SAT, GEKKO, CPLEX
####################################################################################################
####################################################################################################
# Обёртка описания задачи MIP в виде pyomo
####################################################################################################
class pyomo_MIP_model_wrapper:
	def __init__(
		self,
		pyomo,                         # Объект pyomo.environ
		pyomo_MIP_model,               # Модель MIP pyomo.ConcreteModel() со всеми переменными решения и только ограничениями и/или функцией цели, которые могут быть описаны символьно в pyomo
		mip_solver_name="cbc",         # MIP солвер (может быть MILP или MINLP, умеющий работать с классом задач, описанным в pyomo)
		mip_solver_options=None
	):
		self.__pyomo = pyomo
		self.__pyomo_MIP_model = copy.deepcopy(pyomo_MIP_model)
		self.__mip_solver_name = mip_solver_name

		# Нелинейные ограничения (пополняются для каждой итерации, на которой получаются недопустимые значения)
		self.__pyomo_MIP_model.__non_lin_cons = self.__pyomo.ConstraintList()
		# ограничения на функцию цели (линейная аппроксимация функции цели, добавляется после каждой итерации)
		self.__pyomo_MIP_model.__obj_cons = self.__pyomo.ConstraintList()
		# дополнительные пользовательские ограничения
		self.__pyomo_MIP_model.__custom_cons = self.__pyomo.ConstraintList()

		"""
		Если функция цели определена в pyomo, то предполагается, что она линейная, и мы её минимизируем MIP-солвером.
		Если функции цели в pyomo_MILP_model нет, то значит она нелинейная и задана внешний функцией non_lin_obj_fun.
		В этом случае мы будем минимизировать её линейные аппроксимации, и для этого нам понадобится вспомогательная
		переменная решений mu
		"""
		self.__if_objective_defined = (self.__pyomo_MIP_model.nobjectives() > 0)
		if not self.__if_objective_defined:
			# Дополнительная переменная решений - верхняя граница цели (максимум от линейных аппроксимаций)
			self.__pyomo_MIP_model.__mu = self.__pyomo.Var(domain=self.__pyomo.Reals)
			# На первом шаге накладываем на неё ограничение снизу, чтобы было решение
			self.__pyomo_MIP_model.__mu_temp_cons = self.__pyomo.Constraint(expr=self.__pyomo_MIP_model.__mu >= -int(1e9))
			# цель MILP
			self.__pyomo_MIP_model.obj = self.__pyomo.Objective(expr=self.__pyomo_MIP_model.__mu, sense=self.__pyomo.minimize)

		self.__mip_solver = self.__pyomo.SolverFactory(self.__mip_solver_name)
		if mip_solver_options != None:
			for key in mip_solver_options.keys():
				self.__mip_solver.options[key] = mip_solver_options[key]

	# возвращаем модель
	def get_mip_model(self):
		return self.__pyomo_MIP_model

	# возвращаем число аппроксимаций функции цели
	def get_object_cuts_num(self):
		return len(self.__pyomo_MIP_model.__obj_cons)

	# возвращаем число аппроксимаций нелинейных ограничений
	def get_non_lin_constr_cuts_num(self):
		return len(self.__pyomo_MIP_model.__non_lin_cons)

	# удаляем временные ограничения
	def del_temp_constr(self):
		for c in self.__pyomo_MIP_model.component_objects():
			if str(c) == "_pyomo_MIP_model_wrapper__mu_temp_cons":
				self.__pyomo_MIP_model.del_component(self.__pyomo_MIP_model.__mu_temp_cons)
				print("Deleting temporary constraints..")
				return True
		return False

	# задана ли функция цели в pyomo
	def if_objective_defined(self):
		return self.__if_objective_defined

	# значение целевой функции
	def get_objective_value(self):
		if not self.__if_objective_defined:
			return self.__pyomo.value(self.__pyomo_MIP_model.obj)
		else:
			fx = []
			for v in self.__pyomo_MIP_model.component_data_objects(self.__pyomo.Objective):
				fx.append(self.__pyomo.value(v))
			return fx[0]

	# значения переменных решения
	def get_values(self, xvars):
		return list(map(self.__pyomo.value, xvars))

	# очищаем аппроксимационные и пользовательские ограничения
	def clear(self):
		self.__pyomo_MIP_model.__non_lin_cons.clear()
		self.__pyomo_MIP_model.__obj_cons.clear()
		self.__pyomo_MIP_model.__custom_cons.clear()
		if not self.__if_objective_defined:
			# На первом шаге накладываем на вспомогательную переменную решений ограничение снизу, чтобы было решение
			self.del_temp_constr()
			self.__pyomo_MIP_model.__mu_temp_cons = self.__pyomo.Constraint(expr=self.__pyomo_MIP_model.__mu >= -int(1e9))

	# добавляем линеаризованные ограничения на функцию цели
	def add_obj_constr(self, fx, gradf, xgradf, xvars):
		expr = \
			fx - \
			xgradf + \
			sum(xvars[i] * gradf[i] for i in range(len(xvars))) <= self.__pyomo_MIP_model.__mu
		self.__pyomo_MIP_model.__obj_cons.add(expr)

	# добавляем лианеризованные ограничения на нарушенные ограничения
	def add_non_lin_constr(self, k, gx_violated, gradg_violated, xgradg_violated, xvars):
		expr = \
			gx_violated[k] - \
			xgradg_violated[k] + \
			sum(xvars[i] * gradg_violated[k][i] for i in range(len(xvars))) <= 0
		self.__pyomo_MIP_model.__non_lin_cons.add(expr)

	# добавляем дополнительное пользовательское ограничение
	def add_custom_constr(self, expr):
		self.__pyomo_MIP_model.__custom_cons.add(expr)

	def solve(self):
		if self.__mip_solver_name.upper() in ["CBC", "CPLEX"]:
			print("warm start")
			results = self.__mip_solver.solve(self.__pyomo_MIP_model, warmstart=True, tee=False)
		else:
			results = self.__mip_solver.solve(self.__pyomo_MIP_model, tee=False)
		if results.Solver()["Termination condition"] == self.__pyomo.TerminationCondition.infeasible:
			return False
		return True

####################################################################################################
# Обёртка описания задачи MIP в виде MIP
####################################################################################################
class mip_MIP_model_wrapper:
	def __init__(
		self,
		mip_object,                     # mip из import mip as mip
		mip_MIP_model,                  # Модель MIP mip.Model()
		mip_solver_options=None         # TO DO !!!!!!!!!!!!
	):
		self.__mip_object = mip_object
		# copy.deepcopy НЕ РАБОТАТ!!!!!!
		# self.__mip_MIP_model = copy.deepcopy(mip_MIP_model)
		self.__mip_MIP_model = mip_MIP_model

		# Нелинейные ограничения (пополняются для каждой итерации, на которой получаются недопустимые значения)
		self.__mip_MIP_model.__non_lin_cons = []
		# ограничения на функцию цели (линейная аппроксимация функции цели, добавляется после каждой итерации)
		self.__mip_MIP_model.__obj_cons = []
		# дополнительные пользовательские ограничения
		self.__mip_MIP_model.__custom_cons = []

		"""
		Если функция цели определена в pyomo, то предполагается, что она линейная, и мы её минимизируем MIP-солвером.
		Если функции цели в pyomo_MILP_model нет, то значит она нелинейная и задана внешний функцией non_lin_obj_fun.
		В этом случае мы будем минимизировать её линейные аппроксимации, и для этого нам понадобится вспомогательная
		переменная решений mu
		"""
		self.__if_objective_defined = (len(self.__mip_MIP_model.objective.expr.values()) > 0)
		if not self.__if_objective_defined:
			# Дополнительная переменная решений - верхняя граница цели (максимум от линейных аппроксимаций)
			self.__mip_MIP_model.__mu = self.__mip_MIP_model.add_var(name="mu", lb=-np.Inf, ub=np.Inf, var_type="C")
			# На первом шаге накладываем на неё ограничение снизу, чтобы было решение
			self.__mip_MIP_model.__mu_temp_cons = self.__mip_MIP_model.add_constr(self.__mip_MIP_model.__mu >= -1e9)
			# цель MILP
			self.__mip_MIP_model.objective = self.__mip_object.minimize(self.__mip_MIP_model.__mu)
		else:
			self.__mip_MIP_model.__mu_temp_cons = None
			
		if mip_solver_options != None:
			for key in mip_solver_options.keys():
				# добавить параметры оптимизации !!!
				pass

	# возвращаем модель
	def get_mip_model(self):
		return self.__mip_MIP_model

	# возвращаем число аппроксимаций функции цели
	def get_object_cuts_num(self):
		return len(self.__mip_MIP_model.__obj_cons)

	# возвращаем число аппроксимаций нелинейных ограничений
	def get_non_lin_constr_cuts_num(self):
		return len(self.__mip_MIP_model.__non_lin_cons)

	# удаляем временные ограничения
	def del_temp_constr(self):
		if (not self.__if_objective_defined) and (not (self.__mip_MIP_model.__mu_temp_cons is None)):
			print("Deletint temp constraints!")
			self.__mip_MIP_model.remove(self.__mip_MIP_model.__mu_temp_cons)
			self.__mip_MIP_model.__mu_temp_cons = None
			return True
		return False

	# задана ли функция цели в pyomo
	def if_objective_defined(self):
		return self.__if_objective_defined

	# значение целевой функции
	def get_objective_value(self):
		return self.__mip_MIP_model.objective_value

	# значения переменных решения
	def get_values(self, xvars):
		return [x.x for x in xvars]

	# очищаем аппроксимационные и пользовательские ограничения
	def clear(self):
		self.__mip_MIP_model.remove(self.__mip_MIP_model.__non_lin_cons)
		self.__mip_MIP_model.remove(self.__mip_MIP_model.__obj_cons)
		self.__mip_MIP_model.remove(self.__mip_MIP_model.__custom_cons)
		self.__mip_MIP_model.__non_lin_cons.clear()
		self.__mip_MIP_model.__obj_cons.clear()
		self.__mip_MIP_model.__custom_cons.clear()
		if (not self.__if_objective_defined) and (self.__mip_MIP_model.__mu_temp_cons is None):
			# На первом шаге накладываем на вспомогательную переменную решений ограничение снизу, чтобы было решение
			self.__mip_MIP_model.__mu_temp_cons = self.__mip_MIP_model.add_constr(self.__mip_MIP_model.__mu >= -1e9)

	# добавляем линеаризованные ограничения на функцию цели
	def add_obj_constr(self, fx, gradf, xgradf, xvars):
		expr = \
			fx - \
			xgradf + \
			sum(xvars[i] * gradf[i] for i in range(len(xvars))) <= self.__mip_MIP_model.__mu
		new_cons = self.__mip_MIP_model.add_constr(expr)
		self.__mip_MIP_model.__obj_cons.append(new_cons)

	# добавляем лианеризованные ограничения на нарушенные ограничения
	def add_non_lin_constr(self, k, gx_violated, gradg_violated, xgradg_violated, xvars):
		expr = \
			gx_violated[k] - \
			xgradg_violated[k] + \
			sum(xvars[i] * gradg_violated[k][i] for i in range(len(xvars))) <= 0
		new_cons = self.__mip_MIP_model.add_constr(expr)
		self.__mip_MIP_model.__non_lin_cons.append(new_cons)

	# добавляем дополнительное пользовательское ограничение
	def add_custom_constr(self, expr):
		new_cons = self.__mip_MIP_model.add_constr(expr)
		self.__mip_MIP_model.__custom_cons.append(new_cons)

	def solve(self):
		res = self.__mip_MIP_model.optimize()
		if res.value == res.INFEASIBLE:
			return False
		return True
	
####################################################################################################
# Обёртка описания задачи MIP в виде google ortools cp_sat
####################################################################################################
class ortools_cp_sat_MIP_model_wrapper:
	def __init__(
		self,
		ortools_cp_model,              # Объект ortools.sat.python.cp_model
		model_milp_cpsat,              # Модель MIP ortools_cp_model.CpModel() со всеми переменными решения и только ограничениями и/или функцией цели, которые могут быть описаны символьно
		BIG_MULT = 1e6                 # На что умножаем коэффициенты ограничений, чтобы они стали целыми
	):
		self.__ortools_cp_model = ortools_cp_model
		self.__model_milp_cpsat = copy.deepcopy(model_milp_cpsat)
		self.__BIG_MULT = BIG_MULT

		# Нелинейные ограничения (пополняются для каждой итерации, на которой получаются недопустимые значения)
		self.__non_lin_cons = []
		# ограничения на функцию цели (линейная аппроксимация после каждой успешной итерации)
		self.__obj_cons = []
		# пользовательские ограничения (для разбиения допустимого множества)
		self.__custom_cons = []

		"""
		Если функция цели определена в pyomo, то предполагается, что она линейная, и мы её минимизируем MIP-солвером.
		Если функции цели в pyomo_MILP_model нет, то значит она нелинейная и задана внешний функцией non_lin_obj_fun.
		В этом случае мы будем минимизировать её линейные аппроксимации.
		"""
		self.__if_objective_defined = self.__model_milp_cpsat.HasObjective()

		self.__model_milp_cpsat_solver = self.__ortools_cp_model.CpSolver()
		self.__model_milp_cpsat_solver.parameters.max_time_in_seconds = 60.0
		if not self.__if_objective_defined:
			# Дополнительная переменная решений - верхняя граница цели (максимум от линейных аппроксимаций)
			self.__mu = self.__model_milp_cpsat.NewIntVar(-int(1e9), int(1e9), "mu")
			# Добавляем цель
			self.__model_milp_cpsat.Minimize(self.__mu)

	# возвращаем модель
	def get_mip_model(self):
		return self.__model_milp_cpsat

	# возвращаем число аппроксимаций функции цели
	def get_object_cuts_num(self):
		return len(self.__obj_cons)

	# возвращаем число аппроксимаций нелинейных ограничений
	def get_non_lin_constr_cuts_num(self):
		return len(self.__non_lin_cons)

	# удаляем временные ограничения
	def del_temp_constr(self):
		return False

	# задана ли функция цели в pyomo
	def if_objective_defined(self):
		return self.__if_objective_defined

	# значение целевой функции
	def get_objective_value(self):
		return self.__model_milp_cpsat_solver.ObjectiveValue()

	# значения переменных решения
	def get_values(self, xvars):
		return list(map(self.__model_milp_cpsat_solver.Value, xvars))

	# очищаем аппроксимационные ограничения
	def clear(self):
		# self.__non_lin_cons.clear()
		# self.__obj_cons.clear()
		# self.__custom_cons.clear()
		return False

	# добавляем линеаризованные ограничения на функцию цели
	def add_obj_constr(self, fx, gradf, xgradf, xvars):
		expr = int(np.round(self.__BIG_MULT*fx)) - \
			int(np.round(self.__BIG_MULT*xgradf)) + \
			sum(xvars[i] * int(np.round(self.__BIG_MULT*gradf[i])) for i in range(len(xvars))) <= int(self.__BIG_MULT)*self.__mu
		new_constr = self.__model_milp_cpsat.Add(expr)
		self.__obj_cons.append(new_constr)

	# добавляем лианеризованные ограничения на нарушенные ограничения
	def add_non_lin_constr(self, k, gx_violated, gradg_violated, xgradg_violated, xvars):
		expr = int(np.round(self.__BIG_MULT*gx_violated[k])) - \
			int(np.round(self.__BIG_MULT*xgradg_violated[k])) + \
			sum(xvars[i] * int(np.round(self.__BIG_MULT*gradg_violated[k][i])) for i in range(len(xvars))) <= 0
		new_constr = self.__model_milp_cpsat.Add(expr)
		self.__non_lin_cons.append(new_constr)

	# добавляем пользовательские ограничения
	def add_custom_constr(self, expr):
		new_constr = self.__model_milp_cpsat.Add(expr)
		self.__custom_cons.append(new_constr)

	def solve(self):
		results = self.__model_milp_cpsat_solver.Solve(self.__model_milp_cpsat)
		if self.__model_milp_cpsat_solver.StatusName() == 'INFEASIBLE':
			return False
		return True

####################################################################################################
# Обёртка описания задачи MIP в виде google ortools linear solver
####################################################################################################
class ortools_linear_solver_MIP_model_wrapper:
	def __init__(
		self,
		milp_ortools_solver,                # Объект ortools.linear_solver.pywraplp.Solver.CreateSolver('SCIP')
		if_objective_defined=False          # Задана ли цель в задаче
	):
		self.__milp_ortools_solver = milp_ortools_solver # copy.deepcopy не работает!

		# Нелинейные ограничения (пополняются для каждой итерации, на которой получаются недопустимые значения)
		self.__non_lin_cons = []
		# ограничения на функцию цели (линейная аппроксимация после каждой успешной итерации)
		self.__obj_cons = []
		# пользовательские ограничения (для разбиения допустимого множества)
		self.__custom_cons = []

		"""
		Если функция цели определена в pyomo, то предполагается, что она линейная, и мы её минимизируем MIP-солвером.
		Если функции цели в pyomo_MILP_model нет, то значит она нелинейная и задана внешний функцией non_lin_obj_fun.
		В этом случае мы будем минимизировать её линейные аппроксимации.
		"""
		self.__if_objective_defined = if_objective_defined

		if not self.__if_objective_defined:
			# Дополнительная переменная решений - верхняя граница цели (максимум от линейных аппроксимаций)
			self.__mu = self.__milp_ortools_solver.NumVar(-self.__milp_ortools_solver.infinity(), self.__milp_ortools_solver.infinity(), "mu")
			# Добавляем временное ограничение на цель
			self.__temp_mu = self.__milp_ortools_solver.Add(self.__mu >= -1e9)
			# Добавляем цель
			self.__milp_ortools_solver.Minimize(self.__mu)
		else:
			self.__temp_mu = None

	# возвращаем модель
	def get_mip_model(self):
		return self.__milp_ortools_solver

	# возвращаем число аппроксимаций функции цели
	def get_object_cuts_num(self):
		return len(self.__obj_cons)

	# возвращаем число аппроксимаций нелинейных ограничений
	def get_non_lin_constr_cuts_num(self):
		return len(self.__non_lin_cons)

	# удаляем временные ограничения
	def del_temp_constr(self):
		# if (not self.__if_objective_defined) and (not (self.__temp_mu is None)):
		# 	self.__milp_ortools_solver.constraints().remove(self.__temp_mu)
		# 	self.__temp_mu = None
		# 	print("Удаляем временное ограничение.")
		# 	return True
		return False

	# задана ли функция цели в pyomo
	def if_objective_defined(self):
		return self.__if_objective_defined

	# значение целевой функции
	def get_objective_value(self):
		return self.__milp_ortools_solver.Objective().Value()

	# значения переменных решения
	def get_values(self, xvars):
		return [x.SolutionValue() for x in xvars]

	# очищаем аппроксимационные ограничения
	def clear(self):
		# for cons in self.__non_lin_cons:
		# 	self.__milp_ortools_solver.constraints().remove(cons)
		# for cons in self.__obj_cons:
		# 	self.__milp_ortools_solver.constraints().remove(cons)
		# for cons in self.__custom_cons:
		# 	self.__milp_ortools_solver.constraints().remove(cons)
		# self.__non_lin_cons.clear()
		# self.__obj_cons.clear()
		# self.__custom_cons.clear()
		return False

	# добавляем линеаризованные ограничения на функцию цели
	def add_obj_constr(self, fx, gradf, xgradf, xvars):
		expr = fx - \
			xgradf + \
			sum(xvars[i] * gradf[i] for i in range(len(xvars))) <= self.__mu
		new_constr = self.__milp_ortools_solver.Add(expr)
		self.__obj_cons.append(new_constr)

	# добавляем лианеризованные ограничения на нарушенные ограничения
	def add_non_lin_constr(self, k, gx_violated, gradg_violated, xgradg_violated, xvars):
		expr = gx_violated[k] - \
			xgradg_violated[k] + \
			sum(xvars[i] * gradg_violated[k][i] for i in range(len(xvars))) <= 0
		new_constr = self.__milp_ortools_solver.Add(expr)
		self.__non_lin_cons.append(new_constr)

	# добавляем пользовательские ограничения
	def add_custom_constr(self, expr):
		new_constr = self.__milp_ortools_solver.Add(expr)
		self.__custom_cons.append(new_constr)

	def solve(self):
		results = self.__milp_ortools_solver.Solve()
		if results == self.__milp_ortools_solver.INFEASIBLE:
			return False
		return True
	
####################################################################################################
# Обёртка описания задачи MIP в виде CPLEX MP или CP
####################################################################################################
class cplex_MIP_model_wrapper:
	def __init__(
		self,
		model_cplex              # Модель MIP docplex.mp.model.Model() или docplex.cp.model.CpoModel() со всеми переменными решения и только ограничениями и/или функцией цели, которые могут быть описаны символьно
	):
		self.__model_type = "cp" if model_cplex.__class__.__name__ == "CpoModel" else "mp"
		print("model type: " + self.__model_type)
		self.__model_cplex = copy.deepcopy(model_cplex)

		# Нелинейные ограничения (пополняются для каждой итерации, на которой получаются недопустимые значения)
		self.__non_lin_cons = []
		# ограничения на функцию цели (линейная аппроксимация после каждой итерации)
		self.__obj_cons = []
		# дополнительные ограничения (для работы с подмножеством допустимого множества)
		self.__custom_cons = []

		"""
		Если функция цели определена в pyomo, то предполагается, что она линейная, и мы её минимизируем MIP-солвером.
		Если функции цели в pyomo_MILP_model нет, то значит она нелинейная и задана внешний функцией non_lin_obj_fun.
		В этом случае мы будем минимизировать её линейные аппроксимации.
		"""
		self.__if_objective_defined = self.__model_cplex.has_objective() if self.__model_type == "mp" else str(self.__model_cplex.get_objective()) != "None"
		print ("has objective: " + str(self.__if_objective_defined))
		# Добавляем цель
		if not self.__if_objective_defined:
			# Дополнительная переменная решений - верхняя граница цели (максимум от линейных аппроксимаций)
			if self.__model_type == "mp":
				self.__mu = self.__model_cplex.var(lb=-np.inf, ub=np.inf, vartype=self.__model_cplex.continuous_vartype, name="mu")
				# Временное ограничение снизу
				self.__temp_mu = self.__model_cplex.add(self.__mu >= -int(1e9))
			elif self.__model_type == "cp":
				self.__mu = self.__model_cplex.integer_var(min=-int(1e9), max=int(1e9), name="mu")
			# цель
			self.__model_cplex.minimize(self.__mu)

	# возвращаем модель
	def get_mip_model(self):
		return self.__model_cplex

	# возвращаем число аппроксимаций функции цели
	def get_object_cuts_num(self):
		return len(self.__obj_cons)

	# возвращаем число аппроксимаций нелинейных ограничений
	def get_non_lin_constr_cuts_num(self):
		return len(self.__non_lin_cons)

	# удаляем временные ограничения
	def del_temp_constr(self):
		if not self.__if_objective_defined:
			if self.__model_type == "mp":
				# Удаляем временное ограничение
				self.__model_cplex.remove_constraints([self.__temp_mu])
				print("Deleting temporary constraints..")
				return True
		return False

	# задана ли функция цели в pyomo
	def if_objective_defined(self):
		return self.__if_objective_defined

	# значение целевой функции
	def get_objective_value(self):
		if self.__model_type == "mp":
			return self.__model_cplex.objective_value
		else:
			return self.__results.get_objective_value()

	# значения переменных решения
	def get_values(self, xvars):
		if self.__model_type == "mp":
			return [v.solution_value for v in xvars]
		else:
			return [self.__results[v.name] for v in xvars]

	# очищаем аппроксимационные ограничения
	def clear(self):
		if self.__model_type == "mp":
			self.__model_cplex.remove_constraints(self.__non_lin_cons)
			self.__model_cplex.remove_constraints(self.__obj_cons)
			self.__model_cplex.remove_constraints(self.__custom_cons)
			self.__non_lin_cons.clear()
			self.__obj_cons.clear()
			self.__custom_cons.clear()
			if not self.__if_objective_defined:
				# Временное ограничение снизу
				self.__temp_mu = self.__model_cplex.add(self.__mu >= -int(1e9))

	# добавляем линеаризованные ограничения на функцию цели
	def add_obj_constr(self, fx, gradf, xgradf, xvars):
		expr = fx - \
			xgradf + \
			sum(xvars[i] * gradf[i] for i in range(len(xvars))) <= self.__mu
		new_constr = self.__model_cplex.add(expr)
		self.__obj_cons.append(new_constr)

	# добавляем лианеризованные ограничения на нарушенные ограничения
	def add_non_lin_constr(self, k, gx_violated, gradg_violated, xgradg_violated, xvars):
		expr = gx_violated[k] - \
			xgradg_violated[k] + \
			sum(xvars[i] * gradg_violated[k][i] for i in range(len(xvars))) <= 0
		new_constr = self.__model_cplex.add(expr)
		self.__non_lin_cons.append(new_constr)

	# добавляем дополнительное пользовательское ограничение
	def add_custom_constr(self, expr):
		cust_constr = self.__model_cplex.add(expr)
		self.__custom_cons.append(cust_constr)

	def solve(self):
		self.__results = self.__model_cplex.solve()
		if not self.__results:
			return False
		return True

####################################################################################################
# Обёртка описания задачи MIP в виде GEKKO
####################################################################################################
class gekko_MIP_model_wrapper:
	def __init__(
		self,
		model_gekko,  # Модель MIP GEKKO() со всеми переменными решения и только ограничениями и/или функцией цели, которые могут быть описаны символьно
		if_objective_defined
	):
		self.__model_gekko = copy.deepcopy(model_gekko)

		# Нелинейные ограничения (пополняются для каждой итерации, на которой получаются недопустимые значения)
		self.__non_lin_cons = []
		# ограничения на функцию цели (линейная аппроксимация после каждой успешной итерации)
		self.__obj_cons = []
		# пользовательские ограничения для сужения допустимой области
		self.__custom_cons = []

		"""
		Если функция цели определена в pyomo, то предполагается, что она линейная, и мы её минимизируем MIP-солвером.
		Если функции цели в pyomo_MILP_model нет, то значит она нелинейная и задана внешний функцией non_lin_obj_fun.
		В этом случае мы будем минимизировать её линейные аппроксимации.
		"""
		self.__if_objective_defined = if_objective_defined
		if not self.__if_objective_defined:
			# Дополнительная переменная решений - верхняя граница цели (максимум от линейных аппроксимаций)
			self.__mu = self.__model_gekko.Var(lb=-1e9, name="mu")
			self.__model_gekko.Obj(self.__mu)

	# возвращаем модель
	def get_mip_model(self):
		return self.__model_gekko

	# возвращаем число аппроксимаций функции цели
	def get_object_cuts_num(self):
		# print([str(c) for c in self.__obj_cons])
		return len(self.__obj_cons)

	# возвращаем число аппроксимаций нелинейных ограничений
	def get_non_lin_constr_cuts_num(self):
		# print([str(c) for c in self.__non_lin_cons])
		return len(self.__non_lin_cons)

	# удаляем временные ограничения
	def del_temp_constr(self):
		return False

	# задана ли функция цели в pyomo
	def if_objective_defined(self):
		return self.__if_objective_defined
	
	# значение целевой функции
	def get_objective_value(self):
		if not self.__if_objective_defined:
			return self.__mu.value[0]
		return self.__model_gekko.options.objfcnval

	# значения переменных решения
	def get_values(self, xvars):
		return [v.value[0] for v in xvars]

	# очищаем аппроксимационные ограничения
	def clear(self):
		# self.__non_lin_cons.clear()
		# self.__obj_cons.clear()
		# self.__custom_cons.clear()
		return False

	# добавляем линеаризованные ограничения на функцию цели
	def add_obj_constr(self, fx, gradf, xgradf, xvars):
		expr = fx - \
			xgradf + \
			sum(xvars[i] * gradf[i] for i in range(len(xvars))) <= self.__mu
		new_constr = self.__model_gekko.Equation(expr)
		self.__obj_cons.append(new_constr)

	# добавляем лианеризованные ограничения на нарушенные ограничения
	def add_non_lin_constr(self, k, gx_violated, gradg_violated, xgradg_violated, xvars):
		expr = gx_violated[k] - \
			xgradg_violated[k] + \
			sum(xvars[i] * gradg_violated[k][i] for i in range(len(xvars))) <= 0
		new_constr = self.__model_gekko.Equation(expr)
		self.__non_lin_cons.append(new_constr)

	# добавляем пользовательские ограничения
	def add_custom_constr(self, expr):
		new_constr = self.__model_gekko.Equation(expr)
		self.__custom_cons.append(new_constr)

	def solve(self):
		results = self.__model_gekko.solve(disp=False)
		if self.__model_gekko.options.SOLVESTATUS != 1:
			return False
		return True

####################################################################################################
# Класс-оркестратор решения
####################################################################################################
class mmaxgon_MINLP_POA:
	def __init__(self,
		eps=1e-6                      # Приращение аргумента для численного дифференцирования
	):
		self.__eps = eps

	# пустая функция ограничений
	def __constr_true(self, x):
		return [-1]

	# градиент для линейной аппроксимации цели
	def __get_linear_appr(self, fun, x):
		return opt.approx_fprime(x, fun, epsilon = self.__eps)

	# якобиан для линейной аппроксимации матрицы ограничений
	def __get_linear_appr_matrix(self, fun, x):
		return opt.slsqp.approx_jacobian(x, fun, epsilon = self.__eps)

	# решаем задачу нелинейной оптимизации последовательностью линейных аппроксимаций
	def solve(
		self,
		MIP_model,                      # Модель MIP со всеми переменными решения и только ограничениями и/или функцией цели, которые могут быть описаны символьно в pyomo
		non_lin_obj_fun,                # Нелинейная функция цели (нельзя описать символьно)
		non_lin_constr_fun,             # Нелинейные ограничения (list) (нельзя описать символьно)
		decision_vars_to_vector_fun,    # Функция, комбинирующая переменные решения Pyomo в list
		tolerance=1e-1,                 # разница между верхней и нижней оценкой оптимальной функции цели
		add_constr="ALL",               # {"ALL", "ONE"} число нарушенных нелинейных ограничений для которых добавляются линейные ограничения
		NLP_refiner_object=None, 		# Объект класса с моделью NLP со всеми нелинейными ограничениями и с функцией цели для уточнения значений непрерывных переменных при фиксации целочисленных
		NLP_projector_object=None,		# Объект класса с моделью NLP со всеми нелинейными ограничениями для проекции недопустимой точки на допустимую область для последующей построении касательной и линейного ограничения
		lower_bound=None,               # Первичная оченка нижней границы решения
		custom_constraints_list=[],     # Список выражений для дополнительных (не заданных в модели) ограничений для данного решения
		approximation_points=[]         # Список точек, в которых будем строиться линейная аппроксимация нелинейной функции целии и нелинейных ограничений до первого MILP-решения
	):
		# если функция нелинейных ограничений не задана, то ей становится функция-заглушка со значением -1 (т.е. всегда допустимо)
		if non_lin_constr_fun == None:
			non_lin_constr_fun = self.__constr_true

		# Очищаем все технические ограничения, которые были добавлены на прошлом прогоне метода
		MIP_model.clear()

		"""
		Добавляем пользовательские ограничения к модели. Например, при реализации Branch and Bound или глобального поиска.
		Применяется если задача невыпуклая, и тогда требуется разбить допустимую область на меньшие фрагменты.
		"""
		for expr in custom_constraints_list:
			MIP_model.add_custom_constr(expr)

		# Инициируем начальные значения
		x_best = None
		goal_best = np.Inf
		upper_bound = np.Inf
		lower_bound = -np.Inf if lower_bound == None else lower_bound
		prev_obj = -np.Inf
		if_first_run = True # признак первого прогона
		iter_num = 0

		# Если функция цели определена в MIP_model, то предполагается, что она линейная, и мы её минимизируем.
		# Если функции цели в MIP_model нет, то значит она нелинейная и задана внешний функцией non_lin_obj_fun.
		# В этом случае мы будем минимизировать её линейные аппроксимации, и для этого нам понадобится вспомогательная
		# переменная решений mu
		if (not MIP_model.if_objective_defined()) and non_lin_obj_fun == None:
			raise ValueError("Не определена функция цели!")
		if MIP_model.if_objective_defined() and non_lin_obj_fun != None:
			raise ValueError("Одновременно определены две функции цели!")
		
		# список всех переменных решений модели
		xvars = decision_vars_to_vector_fun(MIP_model.get_mip_model())
		
		# строим линейные аппроксимации к нелинейной функции цели и нелинейным ограничениям,
		# чтобы получать с самого начала адекватное MILP-решение
		for x_random in approximation_points:
			# Если функция цели нелинейная, то добавляем новую аппроксимацию функции цели в ограничения
			if not MIP_model.if_objective_defined():
				# fx - значение исходной целевой функции
				fx = non_lin_obj_fun(x_random)
				gradf = self.__get_linear_appr(non_lin_obj_fun, x_random)
				xgradf = np.dot(x_random, gradf)
				MIP_model.add_obj_constr(fx, gradf, xgradf, xvars)
			
			# Если есть нелинейные ограничения, то добавляем линейные ограничения в виде касательной к ним
			if not (non_lin_constr_fun is None):
				gx = np.array(non_lin_constr_fun(x_random))
				gradg = self.__get_linear_appr_matrix(non_lin_constr_fun, x_random)
				xgradg = np.matmul(gradg, x_random)
				# добавляем новые аппроксимацию в ограничения
				for k in range(len(gx)):
					MIP_model.add_non_lin_constr(k, gx, gradg, xgradg, xvars)
					
		# MILP-итерации
		while True:
			iter_num += 1
			# Решаем MIP-задачу
			results = MIP_model.solve()

			if not results:
				print("MIP не нашёл допустимого решения")
				return {
					"x": x_best,
					"obj": goal_best,
					"non_lin_constr_num": MIP_model.get_non_lin_constr_cuts_num(),
					"goal_fun_constr_num": MIP_model.get_object_cuts_num(),
					"iter_num": iter_num,
					"upper_bound": upper_bound,
					"lower_bound": lower_bound
				}

			"""
			MILP-решение (даже недопустимое) даёт нижнюю границу
			На каждой итерации Lower_bound увеличивается потому что:
			1. Если добавляется новое ограничение, то допустимая область уменьшается, значит решение (минимум) растёт
			2. Если строится новая касательная к функции цели, то она тоже добавляется в ограничения, что снова уменьшает
			допустимую область и может только повысить минимум.
			"""
			# значение цели вспомогательной задачи
			obj = MIP_model.get_objective_value()
			print("MIP_model.get_objective_value(): " + str(obj))
			
			# obj (целевая функция вспомогательной задачи) может только увеличиваться, если нет - ошибка (может быть нарушение выпуклости задачи)
			if obj < prev_obj:
				raise ValueError("Значение целевой функции вспомогательной задачи не может уменьшаться!")
			prev_obj = obj
			lower_bound = max(lower_bound, obj)

			# переводим переменные решения в вектор
			x = MIP_model.get_values(xvars)
			print(x)

			# уточняем непрерывные переменные решения
			if NLP_refiner_object != None:
				res = NLP_refiner_object.get_solution(x)
				if res["success"] and res["constr_violation"] <= self.__eps:
					x = copy.copy(res["x"])
					print("После уточнения:\r\n")
					print(x)

			# если это первое решение и функция цели не задана в задаче - убираем ограничение на целевую переменную mu
				if if_first_run and (not MIP_model.if_objective_defined()):
					MIP_model.del_temp_constr()
					if_first_run = False

			# Если функция цели нелинейная, то добавляем новую аппроксимацию функции цели в ограничения
			if not MIP_model.if_objective_defined():
				# fx - значение исходной целевой функции
				fx = non_lin_obj_fun(x)
				gradf = self.__get_linear_appr(non_lin_obj_fun, x)
				xgradf = np.dot(x, gradf)
				MIP_model.add_obj_constr(fx, gradf, xgradf, xvars)
			else:
				# fx - целевая функция, заданная в вспомогательной задаче
				fx = obj

			# получаем индексы для нарушенных нелинейных ограничений
			gx = non_lin_constr_fun(x)
			ix_violated = list(np.where(np.array(gx) > 0)[0])

			# если решение допустимо
			if len(ix_violated) == 0:
				print("Feasible")
				
				# обновляем лучшее решение
				if fx < goal_best:
					x_best = copy.copy(x)
					goal_best = fx
					# верхняя граница - пока самое лучшее решение
					upper_bound = goal_best
					print("Лучшее решение: {0}, {1}".format(x_best, goal_best))

				# сравниваем нижние и верхние границы функции цели
				print("Нижняя и верхняя граница: {0}, {1}".format(lower_bound, upper_bound))
				if upper_bound + self.__eps < lower_bound:
					raise ValueError("upper_bound < lower_bound!")
				if (upper_bound - lower_bound <= tolerance):
					return {
						"x": x_best,
						"obj": goal_best,
						"non_lin_constr_num": MIP_model.get_non_lin_constr_cuts_num(),
						"goal_fun_constr_num": MIP_model.get_object_cuts_num(),
						"iter_num": iter_num,
						"upper_bound": upper_bound,
						"lower_bound": lower_bound
					}
			else:
				print("Not feasible")
				# значения нарушенных ограничений
				gx_violated = np.array(gx)[ix_violated]
				if_recalc = False
				# если нарушенных ограничений несколько, а мы должны выбрать только одно
				if gx_violated.shape[0] > 1 and add_constr == "ONE":
					# оставляем только индекс с максимальным нарушением ограничения
					ix_most_violated = np.argmax(gx_violated)
					ix_violated = [ix_violated[ix_most_violated]]
					if_recalc = True    # поменялось множество индексов ix_violated

				if NLP_projector_object != None:
					res = NLP_projector_object.get_solution(x)
					if res["success"] and res["constr_violation"] <= self.__eps:
						x = copy.copy(res["x"])
						print("После проецирования:\r\n")
						print(x)
						gx = non_lin_constr_fun(x)
						if_recalc = True    # поменялись значения ограничений gx

				if if_recalc:
					gx_violated = np.array(gx)[ix_violated]
				# добавляем линеаризацию нарушенных нелинейных ограничений
				gradg_violated = self.__get_linear_appr_matrix(lambda x: np.array(non_lin_constr_fun(x))[ix_violated], x)
				xgradg_violated = np.matmul(gradg_violated, x)
				# добавляем новые аппроксимацию в ограничения
				for k in range(len(ix_violated)):
					MIP_model.add_non_lin_constr(k, gx_violated, gradg_violated, xgradg_violated, xvars)

###############################################################################

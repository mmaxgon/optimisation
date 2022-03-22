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
objective = namedtuple("objective", ["n", "if_linear", "fun"])
# Линейные ограничения
linear_constraints = namedtuple("linear_constraints", ["m", "A", "bounds"])
# Нелинейные ограничения
nonlinear_constraints = namedtuple("nonlinear_constraints", ["m", "fun", "bounds"])
# Описание оптимизационной задачи
optimization_problem = namedtuple("optimization_problem", ["dvars", "objective", "linear_constraints", "nonlinear_constraints"])

####################################################################################################
# Получение нижней границы решения
####################################################################################################

def get_NLP_lower_bound(opt_prob, custom_linear_constraints = None, custom_nonlinear_constraints = None):
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
5. GOTO 2.
"""
def get_feasible_solution1(opt_prob, x_nlp):
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
	init_bounds_lb = np.array(opt_prob.dvars.bounds.lb)
	init_bounds_ub = np.array(opt_prob.dvars.bounds.ub)
	ix_int = copy.copy(opt_prob.dvars.ix_int)
	x_nlp = np.array(x_nlp)

	while True:
		# значения переменных решения, которые должны быть целочисленными
		x_int = x_nlp[ix_int]
		# Отклонение значений от целочисленных
		x_dist_int = np.amin([np.ceil(x_int) - x_int, x_int - np.floor(x_int)], axis=0)
		# Берём минимальное
		ix = np.argmin(x_dist_int)
		# округляем
		round_val = np.round(x_nlp[ix_int][ix])
		# Фиксируем значение этой переменной
		init_bounds_lb[ix_int[ix]] = round_val
		init_bounds_ub[ix_int[ix]] = round_val
		# Получаем новое NLP-решение
		res = opt.minimize(
			fun=opt_prob.objective.fun,
			bounds=opt.Bounds(init_bounds_lb, init_bounds_ub),
			constraints=constraints,
			x0=x_nlp,
			method="trust-constr",
			options={'verbose': 0, "maxiter": 200}
		)
		if (not res.success) or (res.constr_violation > 1e-6):
			raise ValueError("Feasible solution not found!")

		res = {"x": res.x, "obj": res.fun, "success": res.success, "constr_violation": res.constr_violation}
		print(res)
		# не осталось больше целочисленных компонент, которые мы ещё не зафикисировали
		if len(ix_int) <= 1:
			return res
		x_nlp = res["x"]
		# удаляем из целочисленных индексов уже зафиксированный
		ix_int = np.delete(ix_int, ix)


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
def get_minlp_solution(opt_prob, obj_tolerance=1e-6):
	def if_nonlinconstr_sutisffied(x_milp):
		if opt_prob.nonlinear_constraints == None:
			return True
		if np.all(np.array(opt_prob.nonlinear_constraints.fun(x_milp)) <= np.array(opt_prob.nonlinear_constraints.bounds.ub)) and \
				np.all(np.array(opt_prob.nonlinear_constraints.fun(x_milp)) >= np.array(opt_prob.nonlinear_constraints.bounds.lb)):
			return True
		return False
	
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
	# ограничения при линеаризации нелинейных ограничений
	model_milp.non_lin_cons = pyomo.ConstraintList()
	# ограничения при линеаризации функции цели
	model_milp.obj_cons = pyomo.ConstraintList()
	# вспомогательная функция цели
	model_milp.mu = pyomo.Var(domain=pyomo.Reals)
	if opt_prob.objective.if_linear:
		model_milp.obj_cons.add(expr=opt_prob.objective.fun(x_var) <= model_milp.mu)
	else:
		model_milp.temp_mu = pyomo.Constraint(expr=model_milp.mu >= -1e9)
	model_milp.obj = pyomo.Objective(expr=model_milp.mu, sense=pyomo.minimize)
	
	# начальные значения
	lower_bound = -np.Inf
	upper_bound = np.Inf
	best_sol = None
	if_first_step = True
	sf = pyomo.SolverFactory("cbc")
	
	while True:
		# MILP итерация решения
		# решаем: находим самое близкое MILP-решение к NLP-решению
		result = sf.solve(model_milp, tee=False, warmstart=True)
		if result.Solver()["Termination condition"] == pyomo.TerminationCondition.infeasible:
			raise ValueError("Не найдено MILP-решение!")
		if if_first_step and not opt_prob.objective.if_linear:
			if_first_step = False
			model_milp.del_component(model_milp.temp_mu)
		# значения переменных решения
		x_milp = list(map(pyomo.value, x_var))
		print("MILP: " + str(x_milp))
		# значение целевой функции вспомогательной задачи
		obj = model_milp.mu.value
		lower_bound = obj
		# значение целевой функции исходной задачи (совпадают в случае линейности цели)
		fx = opt_prob.objective.fun(x_milp)
		# линеаризованное ограничение на функцию цели
		if not opt_prob.objective.if_linear:
			gradf = opt.approx_fprime(x_milp, opt_prob.objective.fun, epsilon=1e-6)
			xgradf = np.dot(x_milp, gradf)
			expr = fx + sum(gradf[j] * (x_var[j] - x_milp[j]) for j in range(opt_prob.dvars.n)) <= model_milp.mu
			# print(expr)
			model_milp.obj_cons.add(expr)
		
		if if_nonlinconstr_sutisffied(x_milp):
			print("feasible")
			if fx < upper_bound:
				upper_bound = fx
				best_sol = copy.copy(x_milp)
				
			if (lower_bound > upper_bound):
				raise ValueError("lower_bound > upper_bound!")
			if (upper_bound - lower_bound < obj_tolerance):
				res = {"obj": upper_bound, "x": best_sol}
				print("lower_bound: {0}, upper_bound: {1}".format(lower_bound, upper_bound))
				return res
		else:
			# ДАЛЕЕ ПРЕДПОЛАГАЕМ, ЧТО НЕЛИНЕЙНЫЕ ОГРАНИЧЕНИЯ ВЫГЛЯДЯТ КАК g(x) <= ub, Т.Е., ВЕРХНЯЯ ГРАНИЦА ub, А НИЖНЕЙ НЕТ
			ix_violated = np.where(np.array(opt_prob.nonlinear_constraints.fun(x_milp)) > opt_prob.nonlinear_constraints.bounds.ub)[0]
			# добавляем линеаризованные ограничения
			gradg = opt.slsqp.approx_jacobian(x_milp, lambda x: np.array(opt_prob.nonlinear_constraints.fun(x))[ix_violated], epsilon=1e-9)
			gx = np.array(opt_prob.nonlinear_constraints.fun(x_milp))[ix_violated]
			for i in range(len(ix_violated)):
				expr = gx + sum(gradg[i][j] * (x_var[j] - x_milp[j]) for j in range(opt_prob.dvars.n)) <= 0
				model_milp.non_lin_cons.add(expr)

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
# Обёртка описания задачи MIP в виде GEKKO -- ПОХОЖЕ НЕ РАБОТАЕТ
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
		custom_constraints_list=[]      # Список выражений для дополнительных (не заданных в модели) ограничений для данного решения
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
			xvars = decision_vars_to_vector_fun(MIP_model.get_mip_model())
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

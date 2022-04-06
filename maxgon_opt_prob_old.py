####################################################################################################
# Описание MINLP-задачи и решения простых проблем:
# - последовательное округление
# - последовательное приближение решений NLP и MILP
# - POA с помощью pyomo и cbc
# - свой BB
####################################################################################################
# from collections import namedtuple
from dataclasses import dataclass, field, replace, astuple, asdict
import itertools
import copy
import numpy as np
import scipy.optimize as opt
import pyomo.environ as pyomo

####################################################################################################
# Объекты абстрактного описания задачи
####################################################################################################
# Два варианта описания: старый через namedtuple и новый через dataclass (python >= 3.7)
# Границы
# bounds = namedtuple("bounds", ["lb", "ub"])
@dataclass
class bounds:
	lb: np.ndarray
	ub: np.ndarray
	def __post_init__(self):
		self.lb = np.array(self.lb)
		self.ub = np.array(self.ub)
		assert(self.lb.shape == self.ub.shape)
		assert(np.all(self.lb <= self.ub))

# Переменные решения
# dvars = namedtuple("dvars", ["n", "ix_int", "ix_cont", "bounds", "x0"])
@dataclass
class dvars:
	n: int
	ix_int:  np.ndarray
	ix_cont: np.ndarray
	bounds: bounds
	x0: np.ndarray = None
	def __post_init__(self):
		self.ix_int = np.array(self.ix_int)
		self.ix_cont = np.array(self.ix_cont)
		self.x0 = np.array(self.x0)
		assert(len(np.intersect1d(self.ix_int, self.ix_cont, assume_unique=True, return_indices=False)) == 0)
		assert(len(np.union1d(self.ix_int, self.ix_cont)) == self.n)
		if not (self.x0 is None):
			assert(len(self.x0) == self.n)
		assert(len(self.bounds.ub) == len(self.bounds.lb) == self.n)

# Цель
# objective = namedtuple("objective", ["n", "if_linear", "fun", "lin_coeffs"])
@dataclass
class objective:
	n: int
	if_linear:  bool
	fun: object = None
	lin_coeffs: np.array = None
	def __post_init__(self):
		if self.if_linear:
			assert(not(self.lin_coeffs is None))
			assert(len(self.lin_coeffs) == self.n)
		else:
			assert(not (self.fun is None))

# Линейные ограничения
# linear_constraints = namedtuple("linear_constraints", ["m", "A", "bounds"])
@dataclass
class linear_constraints:
	n: int
	m: int
	A: np.array
	bounds: bounds
	def __post_init__(self):
		assert(not(self.A is None))
		self.A = np.array(self.A)
		assert(self.A.shape == (self.m, self.n))
		assert(self.bounds.ub.shape == self.bounds.lb.shape == (self.m,))

def join_linear_constraints(lin_cons1, lin_cons2):
	assert (lin_cons1.n == lin_cons2.n)
	A = np.stack((lin_cons1.A, lin_cons2.A)).reshape(lin_cons1.m + lin_cons2.m, lin_cons1.n)
	ub = np.stack((lin_cons1.bounds.ub, lin_cons2.bounds.ub)).reshape(lin_cons1.m + lin_cons2.m, )
	lb = np.stack((lin_cons1.bounds.lb, lin_cons2.bounds.lb)).reshape(lin_cons1.m + lin_cons2.m, )
	new_bounds = bounds(lb, ub)
	new_lin_cons = linear_constraints(lin_cons1.n, lin_cons1.m + lin_cons2.m, A, new_bounds)
	return new_lin_cons

# Нелинейные ограничения
# nonlinear_constraints = namedtuple("nonlinear_constraints", ["m", "fun", "bounds"])
@dataclass
class nonlinear_constraints:
	n: int
	m: int
	fun: object
	bounds: bounds
	def __post_init__(self):
		assert(not(self.fun is None))
		assert(self.bounds.ub.shape == self.bounds.lb.shape == (self.m,))

# Описание оптимизационной задачи
# optimization_problem = namedtuple("optimization_problem", ["dvars", "objective", "linear_constraints", "nonlinear_constraints"])
@dataclass
class optimization_problem:
	dvars: dvars
	objective: objective
	linear_constraints: linear_constraints
	nonlinear_constraints: nonlinear_constraints
	def __post_init__(self):
		assert(not(self.dvars is None))
		assert(not(self.objective is None) or not(self.linear_constraints is None) or not (self.nonlinear_constraints is None))
		n = self.dvars.n
		assert(n == self.objective.n)
		if not (self.linear_constraints is None):
			assert(n == self.linear_constraints.n)
		if not (self.nonlinear_constraints is None):
			assert(n == self.nonlinear_constraints.n)
		if not (self.objective is None):
			assert(n == self.objective.n)
		
####################################################################################################
# Получение нижней границы решения
####################################################################################################

def get_relaxed_solution(opt_prob, custom_linear_constraints = None, custom_nonlinear_constraints = None):
	if opt_prob.objective.if_linear and (opt_prob.nonlinear_constraints is None) and (custom_nonlinear_constraints is None):
		# линейная задача
		if not(custom_linear_constraints is None):
			lin_cons = join_linear_constraints(opt_prob.linear_constraints, custom_linear_constraints)
		else:
			lin_cons = opt_prob.linear_constraints
		ix_ub = np.where(lin_cons.bounds.lb < lin_cons.bounds.ub)[0]
		if len(ix_ub) == 0:
			A_ub = None
			b_ub = None
		else:
			A_ub = np.array(lin_cons.A)[ix_ub]
			b_ub = np.array(lin_cons.bounds.ub)[ix_ub]
			
		ix_eq = np.where(lin_cons.bounds.lb == lin_cons.bounds.ub)[0]
		if len(ix_eq) == 0:
			A_eq = None
			b_eq = None
		else:
			A_eq = np.array(lin_cons.A)[ix_eq]
			b_eq = np.array(lin_cons.bounds.ub)[ix_eq]
		
		bounds = []
		for i in range(len(opt_prob.dvars.bounds.lb)):
			bounds.append((opt_prob.dvars.bounds.lb[i], opt_prob.dvars.bounds.ub[i]))

		c = opt_prob.objective.lin_coeffs
		
		res = opt.linprog(
			c=c,
			A_ub=A_ub,
			b_ub=b_ub,
			A_eq=A_eq,
			b_eq=b_eq,
			bounds=bounds,
			method='simplex',
			callback=None,
			options=None,
			x0=opt_prob.dvars.x0
		)
		print(res)
		res = {"x": res.x, "obj": res.fun, "success": res.success, "constr_violation": np.max(res.con)}
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
	if not(custom_linear_constraints is None):
		constraints.append(
			opt.LinearConstraint(
				custom_linear_constraints.A,
				custom_linear_constraints.bounds.lb,
				custom_linear_constraints.bounds.ub
			)
		)
	if not(custom_nonlinear_constraints is None):
		constraints.append(
			opt.NonlinearConstraint(
				custom_nonlinear_constraints.fun,
				custom_nonlinear_constraints.bounds.lb,
				custom_nonlinear_constraints.bounds.ub
			)
		)
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
			x0=x0,
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
4. Находим новое NLP-решение по оставшимся нефиксированным
5. Если решение найдено: GOTO 2.
6. Если решение не найдено: округляем в другую сторону и решаем NLP. Если решение найдено: GOTO 2.
7. Если в обе стороны округления решение не найдено, идём по ветке вверх и меняем сторону округления там.
Для бинарных переменных мы обязательно переберём все возможные варианты.
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
	if opt_prob.linear_constraints is not None:
		for j in range(opt_prob.linear_constraints.m):
			expr = np.matmul(opt_prob.linear_constraints.A[j], x_var)
			if opt_prob.linear_constraints.bounds.ub[j] < np.Inf:
				expr1 = expr <= opt_prob.linear_constraints.bounds.ub[j]
				model_milp.lin_cons.add(expr1)
			if opt_prob.linear_constraints.bounds.lb[j] > - np.Inf:
				expr2 = expr >= opt_prob.linear_constraints.bounds.lb[j]
				model_milp.lin_cons.add(expr2)
	# вспомогательные ограничения при линеаризации
	model_milp.non_lin_cons = pyomo.ConstraintList()
	# ограничения цели
	model_milp.mu = pyomo.Var(domain=pyomo.NonNegativeReals)
	model_milp.obj = pyomo.Objective(expr=model_milp.mu, sense=pyomo.minimize)
	model_milp.obj_cons = pyomo.ConstraintList()
	prev_dist = np.Inf

	sf = pyomo.SolverFactory("cbc", executable="C:\\Program Files\\solvers\\CBC\\bin\\cbc.exe")
	# sf = pyomo.SolverFactory("cbc")
	while True:
		# MILP итерация решения
		model_milp.obj_cons.clear()
		# Функция цели - L1-расстояние до NLP-решения
		for i in range(opt_prob.dvars.n):
			model_milp.obj_cons.add(x_var[i] - x_nlp[i] <= model_milp.mu)
			model_milp.obj_cons.add(x_var[i] - x_nlp[i] >= -model_milp.mu)
		# решаем: находим самое близкое MILP-решение к NLP-решению
		result = sf.solve(model_milp, tee=False)
		if result.Solver()["Termination condition"] == pyomo.TerminationCondition.infeasible:
			raise ValueError("Не найдено MILP-решение!")
		x_milp = [pyomo.value(x) for x in x_var]
		print("MILP: " + str(x_milp))
		# Если найденное решение удобвлетворяем нелинейным ограничениям, то возвращаем его
		if (opt_prob.nonlinear_constraints is None) or (
			np.all(np.array(opt_prob.nonlinear_constraints.fun(x_milp)) <= np.array(opt_prob.nonlinear_constraints.bounds.ub)) and \
			np.all(np.array(opt_prob.nonlinear_constraints.fun(x_milp)) >= np.array(opt_prob.nonlinear_constraints.bounds.lb))
		):
			return {"obj": opt_prob.objective.fun(x_milp), "x": x_milp}
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
			expr = gx[i] + sum(gradg[i][j] * (x_var[j] - x_milp[j]) for j in range(opt_prob.dvars.n)) <= 0
			if str(expr) != "True":
				model_milp.non_lin_cons.add(expr)

		# итерация NLP-приближения к MILP-решению
		nlp_projector = NLP_projector.get_solution(x_milp)
		if (not nlp_projector["success"]) or (nlp_projector["constr_violation"] >= 1e-6):
			raise ValueError("Не найдено NLP-решение!!")
		x_nlp = nlp_projector["x"]
		print("NLP: " + str(x_nlp))

####################################################################################################
# Polyhedral Outer Approximation (pyomo+cbc)
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
	
def get_POA_solution(
	opt_prob,                       # описание задачи
	obj_tolerance=1e-6,             # разница между upper_bound и lower_bound
	if_nlp_lower_bound=False,       # нужно ли рассчитывать нижнюю границу NLP-задачи в начале
	if_refine=False,                # нужно ли после каждого MILP-решения фиксировать целочисленные переменные и уточнять непрерывные
	if_project=False,               # нужно ли проецировать недопустимое решение на допустимое множество и строить касательные к нелинейным ограничениям в точке проекции
	random_points_count=0,          # сколько случайных точек сгенерировать вместе с касательными к функции цели и нелинейным ограничениям в них до начала решения MILP-задачи
	cbc_executable_path="C:\\Program Files\\solvers\\CBC\\bin\\cbc.exe"
):
	# добавление линейных ограничений в качестве касательных к нелинейной функции цели в точке x
	def add_obj_constraints(x, dvar_x, fx=None):
		if fx is None:
			fx = opt_prob.objective.fun(x)
		gradf = opt.approx_fprime(x, opt_prob.objective.fun, epsilon=1e-9)
		expr = fx + sum(gradf[j] * (dvar_x[j] - x[j]) for j in range(opt_prob.dvars.n)) <= model_milp.mu
		# print(expr)
		if (str(expr) != "True"):
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
	if opt_prob.linear_constraints is not None:
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
		# должна быть задана линейная функция или список коэффициентов
		assert((not (opt_prob.objective.lin_coeffs is None)) or (not (opt_prob.objective.fun is None)))
		if not (opt_prob.objective.lin_coeffs is None):
			assert(len(opt_prob.objective.lin_coeffs) == len(x_var))
			expr = sum(opt_prob.objective.lin_coeffs[i] * x_var[i] for i in range(len(opt_prob.objective.lin_coeffs))) <= model_milp.mu
		elif not (opt_prob.objective.fun is None):
			expr = opt_prob.objective.fun(x_var) <= model_milp.mu
		else:
			raise ValueError("Не задана функция цели или коэффициенты для линейной задачи")
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
	if cbc_executable_path is None:
		sf = pyomo.SolverFactory("cbc")
	else:
		sf = pyomo.SolverFactory("cbc", executable=cbc_executable_path)

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
# Branch and Bound (scipy.optimize)
####################################################################################################
def get_BB_solution(opt_prob, int_eps=1e-3, upper_bound=np.inf):
	global_vars = {
		"best_sol" : None,
		"best_obj" : upper_bound,
		"bounds_visited" : []
	}

	def if_integer_feasible(opt_prob, x, eps=int_eps):
		if (opt_prob.dvars.ix_int is None) or (len(opt_prob.dvars.ix_int) == 0):
			return True
		x_int = x[opt_prob.dvars.ix_int]
		if np.all(np.min((np.ceil(x_int) - x_int, x_int - np.floor(x_int)), axis=0) <= eps):
			return True
		return False

	def split_optimization_problem(opt_prob, x):
		if if_integer_feasible(opt_prob, x):
			return
		x_int = x[opt_prob.dvars.ix_int]
		# разбиваем по самой "нецелочисленной" переменной
		ix_var = np.argmax(np.min(np.stack((x_int - np.floor(x_int), np.ceil(x_int) - x_int)), axis=0))
		val_var = x_int[ix_var]
		# текущие границы для этой переменной
		lb = opt_prob.dvars.bounds.lb[opt_prob.dvars.ix_int[ix_var]]
		ub = opt_prob.dvars.bounds.ub[opt_prob.dvars.ix_int[ix_var]]
		# задача слева
		lb_left = lb
		ub_left = np.floor(val_var)
		if lb_left <= ub_left:
			bounds_left = bounds(lb=opt_prob.dvars.bounds.lb, ub=opt_prob.dvars.bounds.ub)
			bounds_left.lb[opt_prob.dvars.ix_int[ix_var]] = lb_left
			bounds_left.ub[opt_prob.dvars.ix_int[ix_var]] = ub_left
			dvars_left = dvars(
				n=opt_prob.dvars.n,
				ix_int=opt_prob.dvars.ix_int,
				ix_cont=opt_prob.dvars.ix_cont,
				bounds=bounds_left,
				x0=x
			)
			yield optimization_problem(
				dvars=dvars_left,
				objective=opt_prob.objective,
				linear_constraints=opt_prob.linear_constraints,
				nonlinear_constraints=opt_prob.nonlinear_constraints
			)
		# задача справа
		lb_right = np.ceil(val_var)
		ub_right = ub
		if lb_right <= ub_right:
			bounds_right = bounds(lb=opt_prob.dvars.bounds.lb, ub=opt_prob.dvars.bounds.ub)
			bounds_right.lb[opt_prob.dvars.ix_int[ix_var]] = lb_right
			bounds_right.ub[opt_prob.dvars.ix_int[ix_var]] = ub_right
			dvars_right = dvars(
				n=opt_prob.dvars.n,
				ix_int=opt_prob.dvars.ix_int,
				ix_cont=opt_prob.dvars.ix_cont,
				bounds=bounds_right,
				x0=x
			)
			yield optimization_problem(
				dvars=dvars_right,
				objective=opt_prob.objective,
				linear_constraints=opt_prob.linear_constraints,
				nonlinear_constraints=opt_prob.nonlinear_constraints
			)

	def go_down(new_opt_prob, global_vars):
		# def go_down(*prob_vars):
		# 	new_opt_prob, global_vars = prob_vars[0]
		print(new_opt_prob.dvars.bounds)
		hash_bounds = hash(str(new_opt_prob.dvars.bounds))
		if hash_bounds in global_vars["bounds_visited"]:
			return None
		global_vars["bounds_visited"].append(hash_bounds)
		new_res = get_BB_solution_internal(new_opt_prob, global_vars)
		return new_res
			
	def get_BB_solution_internal(opt_prob, global_vars):
		res = get_relaxed_solution(opt_prob)
		if not(res["success"] and res["constr_violation"] <= 1e-6):
			print("NO NLP")
			return None
		print(res)
		x = res["x"]
		obj = res["obj"]
		if obj >= global_vars["best_obj"]:
			print("SKIP")
			return None
		if if_integer_feasible(opt_prob, x):
			print("feasible")
			if obj < global_vars["best_obj"]:
				global_vars["best_obj"] = res["obj"]
				global_vars["best_sol"] = x
				print("best solution: " + str(global_vars["best_sol"]) + str(global_vars["best_obj"]))
			return res
		
		# new_probs = [(new_opt_prob, global_vars) for new_opt_prob in split_optimization_problem(opt_prob, x)]
		# new_results = list(map(go_down, new_probs))
		new_results = list(map(lambda new_opt_prob: go_down(new_opt_prob, global_vars), split_optimization_problem(opt_prob, x)))
		new_results = [r for r in new_results if r is not None]
		if len(new_results) == 0:
			return None
		elif len(new_results) == 1:
			return new_results[0]
		else:
			return new_results[0] if new_results[0]["obj"] <= new_results[1]["obj"] else new_results[1]

	return get_BB_solution_internal(opt_prob, global_vars)

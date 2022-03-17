####################################################################################################
# MINLP with Polyhedral Outer Approximation
####################################################################################################
from collections import namedtuple
import copy
import numpy as np
import scipy.optimize as opt

# import pyomo.environ as pyomo
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
objective = namedtuple("objective", ["n", "fun"])
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
# Проекция недопустимой точки на допустимую область
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

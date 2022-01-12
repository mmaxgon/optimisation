####################################################################################################
# MINLP with Polyhedral Outer Approximation
####################################################################################################
import copy
from time import time
import numpy as np

import pyomo.environ as pyomo
import scipy.optimize as opt

import ortools.sat.python.cp_model as ortools_cp_model

####################################################################################################
# Обёртка описания задачи MIP в виде pyomo
####################################################################################################
class pyomo_MIP_model_wrapper:
	def __init__(
		self,
		pyomo,                         # Объект pyomo.environ
		pyomo_MIP_model,               # Модель MIP pyomo.ConcreteModel() со всеми переменными решения и только ограничениями и/или функцией цели, которые могут быть описаны символьно в pyomo
		mip_solver_name="cbc"          # MIP солвер (может быть MILP или MINLP, умеющий работать с классом задач, описанным в pyomo)
	):
		self.__pyomo = pyomo
		self.__pyomo_MIP_model = copy.deepcopy(pyomo_MIP_model)
		self.__mip_solver_name = mip_solver_name

		# Нелинейные ограничения (пополняются для каждой итерации, на которой получаются недопустимые значения)
		self.__pyomo_MIP_model.__non_lin_cons = self.__pyomo.ConstraintList()
		# ограничения на функцию цели (линейная аппроксимация после каждой успешной итерации)
		self.__pyomo_MIP_model.__obj_cons = self.__pyomo.ConstraintList()

		"""
		Если функция цели определена в pyomo, то предполагается, что она линейная, и мы её минимизируем MIP-солвером.
		Если функции цели в pyomo_MILP_model нет, то значит она нелинейная и задана внешний функцией non_lin_obj_fun.
		В этом случае мы будем минимизировать её линейные аппроксимации, и для этого нам понадобится вспомогательная
		переменная решений mu
		"""
		self.__if_objective_defined = (self.__pyomo_MIP_model.nobjectives() > 0)
		if not self.__if_objective_defined:
			# Дополнительная переменная решений - верхняя граница цели (максимум от линейных аппроксимаций)
			self.__pyomo_MIP_model.__mu = self.__pyomo.Var(domain=self.__pyomo.Reals, initialize=0)
			# На первом шаге накладываем на неё ограничение снизу, чтобы было решение
			self.__pyomo_MIP_model.__mu_temp_cons = self.__pyomo.Constraint(expr=self.__pyomo_MIP_model.__mu >= -1e6)
			# цель MILP
			self.__pyomo_MIP_model.obj = self.__pyomo.Objective(expr=self.__pyomo_MIP_model.__mu, sense=self.__pyomo.minimize)

		self.__mip_solver = self.__pyomo.SolverFactory(self.__mip_solver_name)

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

	# очищаем аппроксимационные ограничения
	def clear(self):
		self.__pyomo_MIP_model.__non_lin_cons.clear()
		self.__pyomo_MIP_model.__obj_cons.clear()
		if not self.__if_objective_defined:
			# На первом шаге накладываем на вспомогательную переменную решений ограничение снизу, чтобы было решение
			self.del_temp_constr()
			self.__pyomo_MIP_model.__mu_temp_cons = self.__pyomo.Constraint(expr=self.__pyomo_MIP_model.__mu >= -1e6)

	# добавляем лианеризованные ограничения на функцию цели
	def add_obj_constr(self, fx, gradf, xgradf, xvars):
		self.__pyomo_MIP_model.__obj_cons.add(
			fx - \
			xgradf + \
			sum(xvars[i] * gradf[i] for i in range(len(x))) <= self.__pyomo_MIP_model.__mu
		)

	# добавляем лианеризованные ограничения на нарушенные ограничения
	def add_non_lin_constr(self, k, gx_violated, gradg_violated, xgradg_violated, xvars):
		self.__pyomo_MIP_model.__non_lin_cons.add(
			gx_violated[k] - \
			xgradg_violated[k] + \
			sum(xvars[i] * gradg_violated[k][i] for i in range(len(x))) <= 0
		)

	def solve(self):
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

		"""
		Если функция цели определена в pyomo, то предполагается, что она линейная, и мы её минимизируем MIP-солвером.
		Если функции цели в pyomo_MILP_model нет, то значит она нелинейная и задана внешний функцией non_lin_obj_fun.
		В этом случае мы будем минимизировать её линейные аппроксимации.
		"""
		self.__if_objective_defined = self.__model_milp_cpsat.HasObjective()
		if not self.__if_objective_defined:
			# Дополнительная переменная решений - верхняя граница цели (максимум от линейных аппроксимаций)
			self.__mu = self.__model_milp_cpsat.NewIntVar(-1000000000, 1000000000, "mu")
			self.__model_milp_cpsat.Minimize(self.__mu)

		self.__model_milp_cpsat_solver = self.__ortools_cp_model.CpSolver()
		self.__model_milp_cpsat_solver.parameters.max_time_in_seconds = 60.0

	# возвращаем модель
	def get_mip_model(self):
		return self.__model_milp_cpsat

	# возвращаем число аппроксимаций функции цели
	def get_object_cuts_num(self):
		return len(self.__obj_cons)

	# возвращаем число аппроксимаций нелинейных ограничений
	def get_non_lin_constr_cuts_num(self):
		return len(self.__obj_cons)

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
		self.__non_lin_cons.clear()
		self.__obj_cons.clear()

	# добавляем лианеризованные ограничения на функцию цели
	def add_obj_constr(self, fx, gradf, xgradf, xvars):
		expr = int(np.round(self.__BIG_MULT*fx)) - \
			int(np.round(self.__BIG_MULT*xgradf)) + \
			sum(xvars[i] * int(np.round(self.__BIG_MULT*gradf[i])) for i in range(len(x))) <= int(self.__BIG_MULT)*self.__mu
		new_constr = self.__model_milp_cpsat.Add(expr)
		self.__obj_cons.append(new_constr)

	# добавляем лианеризованные ограничения на нарушенные ограничения
	def add_non_lin_constr(self, k, gx_violated, gradg_violated, xgradg_violated, xvars):
		expr = int(np.round(self.__BIG_MULT*gx_violated[k])) - \
			int(np.round(self.__BIG_MULT*xgradg_violated[k])) + \
			sum(xvars[i] * int(np.round(self.__BIG_MULT*gradg_violated[k][i])) for i in range(len(x))) <= 0
		new_constr = self.__model_milp_cpsat.Add(expr)
		self.__non_lin_cons.append(new_constr)

	def solve(self):
		results = self.__model_milp_cpsat_solver.Solve(self.__model_milp_cpsat)
		if self.__model_milp_cpsat_solver.StatusName() == 'INFEASIBLE':
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
		NLP_refiner_class=None, 		# Класс с моделью NLP со всеми нелинейными ограничениями и с функцией цели для уточнения значений непрерывных переменных при фиксации целочисленных
		NLP_projector_object=None		# Объект класса с моделью NLP со всеми нелинейными ограничениями для проекции недопустимой точки на допустимую область для последующей построении касательной и линейного ограничения
	):
		# если функция нелинейных ограничений не задана, то ей становится функция-заглушка со значением -1 (т.е. всегда допустимо)
		if non_lin_constr_fun == None:
			non_lin_constr_fun = self.__constr_true

		x_best = None
		x_feasible = []
		goal_best = np.Inf
		upper_bound = np.Inf
		lower_bound = -np.Inf
		if_first_feasible = True
		iter_num = 0

		MIP_model.clear()

		# Если функция цели определена в MIP_model, то предполагается, что она линейная, и мы её минимизируем.
		# Если функции цели в MIP_model нет, то значит она нелинейная и задана внешний функцией non_lin_obj_fun.
		# В этом случае мы будем минимизировать её линейные аппроксимации, и для этого нам понадобится вспомогательная
		# переменная решений mu
		if not MIP_model.if_objective_defined() and non_lin_obj_fun == None:
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
			lower_bound = MIP_model.get_objective_value()

			# переводим переменные решения в вектор
			xvars = decision_vars_to_vector_fun(MIP_model.get_mip_model())
			x = MIP_model.get_values(xvars)
			print(x)

			# уточняем непрерывные переменные решения
			if NLP_refiner_class != None:
				new_refiner_model = NLP_refiner_class(x)
				res = new_refiner_model.get_solution()
				if res["success"]:
					x = copy.copy(res["x"])
					print("После уточнения:\r\n")
					print(x)

			# если это первое решение - убираем ограничение на целевую переменную mu
			if if_first_feasible and non_lin_obj_fun != None:
				MIP_model.del_temp_constr()
				if_first_feasible = False

			# Если функция цели нелинейная, то добавляем новую аппроксимацию функции цели в ограничения
			if non_lin_obj_fun != None:
				fx = non_lin_obj_fun(x)
				gradf = self.__get_linear_appr(non_lin_obj_fun, x)
				xgradf = np.dot(x, gradf)
				MIP_model.add_obj_constr(fx, gradf, xgradf, xvars)
			else:
				fx = MIP_model.get_objective_value()

			# получаем индексы для нарушенных нелинейных ограничений
			gx = non_lin_constr_fun(x)
			ix_violated = list(np.where(np.array(gx) > 0)[0])

			# если решение допустимо
			if len(ix_violated) == 0:
				print("Feasible")
				# проверяем было ли уже данное решение, если да, то оно - оптимальное решение
				for y in x_feasible:
					if np.allclose(x, y):
						print("All close!")
						return {
							"x": x_best,
							"obj": goal_best,
							"non_lin_constr_num": MIP_model.get_non_lin_constr_cuts_num(),
							"goal_fun_constr_num": MIP_model.get_object_cuts_num(),
							"iter_num": iter_num,
							"upper_bound": upper_bound,
							"lower_bound": lower_bound
						}

				# если функция цели такая же как у лучшего решения - сохраняем это решение
				if np.isclose(fx, goal_best):
					x_feasible.append(x)
				# обновляем лучшее решение
				if fx < goal_best:
					x_best = copy.copy(x)
					goal_best = fx
					# верхняя граница - пока самое лучшее решение
					upper_bound = goal_best
					print(x_best, goal_best)
					# если найдено лучшее решение, то все прошлые лучшие рещения удаляются и добавляется новое
					x_feasible.clear()
					x_feasible.append(x_best)

				# сравниваем нижние и верхние границы функции цели
				print(lower_bound, upper_bound)
				if upper_bound < lower_bound:
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
				# если нарушенных ограничений несколько, а мы должны выбрать только одно
				if gx_violated.shape[0] > 1 and add_constr == "ONE":
					# оставляем только индекс с максимальным нарушением ограничения
					ix_most_violated = np.argmax(gx_violated)
					ix_violated = [ix_violated[ix_most_violated]]

				if NLP_projector_object != None:
					res = NLP_projector_object.get_solution(x)
					if res["success"]:
						x = copy.copy(res["x"])
						print("После проецирования:\r\n")
						print(x)
						gx = non_lin_constr_fun(x)

				gx_violated = np.array(gx)[ix_violated]
				# добавляем линеаризацию нарушенных нелинейных ограничений
				gradg_violated = self.__get_linear_appr_matrix(lambda x: np.array(non_lin_constr_fun(x))[ix_violated], x)
				xgradg_violated = np.matmul(gradg_violated, x)
				# добавляем новые аппроксимацию в ограничения
				for k in range(len(ix_violated)):
					MIP_model.add_non_lin_constr(k, gx_violated, gradg_violated, xgradg_violated, xvars)

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

# result = pyomo.SolverFactory("cbc").solve(model_milp)
# [model_milp.y[0](), model_milp.x[1](), model_milp.x[1]()]

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
model_minlp.obj = pyomo.Objective(expr = -(1000 - model_minlp.y[0]**2 - 2*model_minlp.x[1]**2 - model_minlp.x[2]**2 - model_minlp.y[0]*x[1] - model_minlp.y[0]*model_minlp.x[2]), sense=pyomo.minimize)

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

###############################################################################
# Функция, переводящая переменные решения pyomo в вектор
def DV_2_vec(model):
	x = [model.y[0], model.x[1], model.x[2]]
	return x

# Функция, переводящая переменные решения cp_sat в вектор
def DV_2_vec_cp_sat(model):
	x = [model.GetIntVarFromProtoIndex(0), model.GetIntVarFromProtoIndex(1), model.GetIntVarFromProtoIndex(2)]
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
		self.linear_constraint = opt.LinearConstraint([[8]], [56-14*self.x[0]-7*self.x[1]], [56--14*self.x[0]-7*self.x[1]])
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
poa = mmaxgon_MINLP_POA(
	eps=1e-6
)

###############################################################################
# ortools cp_sat Нелинейная функция цели, есть нелинейные ограничения
###############################################################################
ortools_cp_sat_mip_model_wrapper = ortools_cp_sat_MIP_model_wrapper(
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
pyomo_mip_model_wrapper = pyomo_MIP_model_wrapper(
	pyomo=pyomo,
	pyomo_MIP_model=model_milp,
	mip_solver_name="cbc"
)

start_time = time()
res1 = poa.solve(
	MIP_model=pyomo_mip_model_wrapper,
	non_lin_obj_fun=obj,
	non_lin_constr_fun=non_lin_cons,
	decision_vars_to_vector_fun=DV_2_vec,
	tolerance=1e-1,
	add_constr="ONE",
	# NLP_refiner_class=scipy_refiner_optimizer,
	NLP_projector_object=scipy_projector_optimizer_obj
)
print(time() - start_time)

pyomo_mip_model_wrapper = pyomo_MIP_model_wrapper(
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
pyomo_mip_model_wrapper = pyomo_MIP_model_wrapper(
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

pyomo_mip_model_wrapper = pyomo_MIP_model_wrapper(
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
pyomo_mip_model_wrapper = pyomo_MIP_model_wrapper(
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
pyomo_mip_model_wrapper = pyomo_MIP_model_wrapper(
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
pyomo_mip_model_wrapper = pyomo_MIP_model_wrapper(
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
pyomo_mip_model_wrapper = pyomo_MIP_model_wrapper(
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
pyomo_mip_model_wrapper = pyomo_MIP_model_wrapper(
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


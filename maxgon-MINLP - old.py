##############################################################################
# MINLP with Polyhedral Outer Approximation
##############################################################################
import copy
from time import time
import numpy as np
import pyomo.environ as pyomo
import scipy.optimize as opt

class mmaxgon_MINLP_POA:
	def __init__(self,
		pyomo,                         # Объект Pyomo
		pyomo_MILP_model,              # Модель MILP со всеми переменными решения и только ограничениями и/или функцией цели, которые могут быть описаны символьно в pyomo
		non_lin_obj_fun,               # Нелинейная функция цели (нельзя описать символьно)
		non_lin_constr_fun,            # Нелинейные ограничения (list) (нельзя описать символьно)
		decision_vars_to_vector_fun,   # Функция, комбинирующая переменные решения Pyomo в list
		eps=1e-6,                      # Приращение аргумента для численного дифференцирования
		milp_solver="cbc",             # MIP солвер (может быть MILP или MINLP, умеющий работать с классом задач, описанным в pyomo)
		NLP_refiner_class=None,        # Класс с моделью NLP со всеми нелинейными ограничениями и с функцией цели для уточнения значений непрерывных переменных при фиксации целочисленных
		NLP_projector_object=None      # Объект класса с моделью NLP со всеми нелинейными ограничениями для проекции недопустимой точки на допустимую область для последующей построении касательной и линейного ограничения
	):
		self.__pyomo = pyomo
		self.__pyomo_MILP_model = pyomo_MILP_model
		self.__non_lin_obj_fun = non_lin_obj_fun
		self.__non_lin_constr_fun = non_lin_constr_fun
		self.__decision_vars_to_vector_fun = decision_vars_to_vector_fun
		self.__eps = eps
		self.__milp_solver = milp_solver
		self.__NLP_refiner_class = NLP_refiner_class
		self.__NLP_projector_object = NLP_projector_object
		# если функция нелинейных ограничений не задана, то ей становится функция-заглушка со значением -1 (т.е. всегда допустимо)
		if self.__non_lin_constr_fun == None:
			self.__non_lin_constr_fun = self.__constr_true

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
		tolerance=1e-1,         # разница между верхней и нижней оценкой оптимальной функции цели
		add_constr="ALL",       # {"ALL", "ONE"} число нарушенных нелинейных ограничений для которых добавляются линейные ограничения
		tee = False
	):
		x_best = None
		x_feasible = []
		goal_best = np.Inf
		upper_bound = np.Inf
		lower_bound = -np.Inf
		if_first_feasible = True
		iter_num = 0

		pyomo_MILP_model = copy.deepcopy(self.__pyomo_MILP_model)

		# Нелинейные ограничения (пополняются для каждой итерации, на которой получаются недопустимые значения)
		pyomo_MILP_model.__non_lin_cons = self.__pyomo.ConstraintList()
		# ограничения на функцию цели (линейная аппроксимация после каждой успешной итерации)
		pyomo_MILP_model.__obj_cons = self.__pyomo.ConstraintList()

		# Если функция цели определена в pyomo_MILP_model, то предполагается, что она линейная, и мы её минимизируем.
		# Если функции цели в pyomo_MILP_model нет, то значит она нелинейная и задана внешний функцией non_lin_obj_fun.
		# В этом случае мы будем минимизировать её линейные аппроксимации, и для этого нам понадобится вспомогательная
		# переменная решений mu
		if pyomo_MILP_model.nobjectives() == 0 and self.__non_lin_obj_fun == None:
			raise ValueError("Не определена функция цели!")
		if pyomo_MILP_model.nobjectives() > 0 and self.__non_lin_obj_fun != None:
			raise ValueError("Одновременно определены две функции цели!")

		if pyomo_MILP_model.nobjectives() == 0 and self.__non_lin_obj_fun != None:
			# Дополнительная переменная решений - верхняя граница цели (максимум от линейных аппроксимаций)
			pyomo_MILP_model.__mu = self.__pyomo.Var(domain=self.__pyomo.Reals, initialize=0)
			# На первом шаге накладываем на неё ограничение снизу, чтобы было решение
			pyomo_MILP_model.__mu_temp_cons = self.__pyomo.Constraint(expr=pyomo_MILP_model.__mu >= -1e6)
			# цель MILP
			pyomo_MILP_model.obj = self.__pyomo.Objective(expr=pyomo_MILP_model.__mu, sense=self.__pyomo.minimize)

		# солвер
		milp_solver = self.__pyomo.SolverFactory(self.__milp_solver)

		while True:
			iter_num += 1
			# Решаем MIP-задачу
			results = milp_solver.solve(pyomo_MILP_model, tee = tee)

			if results.Solver()["Termination condition"] == self.__pyomo.TerminationCondition.infeasible:
				print("MILP не нашёл допустимого решения")
				return {
					"x": x_best,
					"obj": goal_best,
					"non_lin_constr_num": len(pyomo_MILP_model.__non_lin_cons),
					"goal_fun_constr_num": len(pyomo_MILP_model.__obj_cons),
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
			lower_bound = self.__pyomo.value(pyomo_MILP_model.obj)

			# переводим переменные решения в вектор
			xvars = self.__decision_vars_to_vector_fun(pyomo_MILP_model)
			x = list(map(self.__pyomo.value, xvars))
			print(x)

			# уточняем непрерывные переменные решения
			if self.__NLP_refiner_class != None:
				new_refiner_model = self.__NLP_refiner_class(x)
				res = new_refiner_model.get_solution()
				if res["success"]:
					x = copy.copy(res["x"])
					print("После уточнения:\r\n")
					print(x)

			# если это первое решение - убираем ограничение на целевую переменную mu
			if if_first_feasible and self.__non_lin_obj_fun != None:
				pyomo_MILP_model.del_component(pyomo_MILP_model.__mu_temp_cons)
				if_first_feasible = False

			# Если функция цели нелинейная, то добавляем новую аппроксимацию функции цели в ограничения
			if self.__non_lin_obj_fun != None:
				fx = self.__non_lin_obj_fun(x)
				gradf = self.__get_linear_appr(self.__non_lin_obj_fun, x)
				xgradf = np.dot(x, gradf)
				pyomo_MILP_model.__obj_cons.add(
					fx - \
					xgradf + \
					sum(xvars[i] * gradf[i] for i in range(len(x))) <= pyomo_MILP_model.__mu
				)
				#pyomo_MILP_model.__obj_cons.pprint()
			else:
				fx = []
				for v in pyomo_MILP_model.component_data_objects(self.__pyomo.Objective):
					fx.append(self.__pyomo.value(v))
				fx = fx[0]

			# получаем индексы для нарушенных нелинейных ограничений
			gx = self.__non_lin_constr_fun(x)
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
							"non_lin_constr_num": len(pyomo_MILP_model.__non_lin_cons),
							"goal_fun_constr_num": len(pyomo_MILP_model.__obj_cons),
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
						"non_lin_constr_num": len(pyomo_MILP_model.__non_lin_cons),
						"goal_fun_constr_num": len(pyomo_MILP_model.__obj_cons),
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

				if self.__NLP_projector_object != None:
					res = self.__NLP_projector_object.get_solution(x)
					if res["success"]:
						x = copy.copy(res["x"])
						print("После проецирования:\r\n")
						print(x)
						gx = self.__non_lin_constr_fun(x)

				gx_violated = np.array(gx)[ix_violated]
				# добавляем линеаризацию нарушенных нелинейных ограничений
				gradg_violated = self.__get_linear_appr_matrix(lambda x: np.array(self.__non_lin_constr_fun(x))[ix_violated], x)
				xgradg_violated = np.matmul(gradg_violated, x)
				# добавляем новые аппроксимацию в ограничения
				for k in range(len(ix_violated)):
					pyomo_MILP_model.__non_lin_cons.add(
						gx_violated[k] - \
						xgradg_violated[k] + \
						sum(xvars[i] * gradg_violated[k][i] for i in range(len(x))) <= 0
					)

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
# Задача как MILP
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

##############################################################################
# Задача NLP - добавляем нелинейные ограничения
##############################################################################
model_nlp = copy.deepcopy(model_milp)
# переменные решения
# model_nlp.x = pyomo.Var([1, 2], domain = pyomo.NonNegativeIntegers, bounds = (0, 10), initialize = init_integer)
# model_nlp.y = pyomo.Var([0], domain = pyomo.NonNegativeReals, bounds = (0, 10), initialize = init_integer)
# Линейные ограничения
# model_nlp.lin_cons = pyomo.Constraint(expr = 8 * model_nlp.y[0] + 14 * model_nlp.x[1] + 7 * model_nlp.x[2] - 56 == 0)
# нелинейные ограничения
model_nlp.nlc1 = pyomo.Constraint(expr = model_nlp.y[0]**2 + model_nlp.x[1]**2 + model_nlp.x[2]**2 <= 25)
model_nlp.nlc2 = pyomo.Constraint(expr = model_nlp.x[1]**2 + model_nlp.x[2]**2 <= 12)
# нелинейная цель
# model_nlp.obj = pyomo.Objective(expr = -(1000 - model_nlp.y[0]**2 - 2*model_nlp.x[1]**2 - model_nlp.x[2]**2 - model_nlp.y[0]*x[1] - model_nlp.y[0]*model_nlp.x[2]), sense=pyomo.minimize)
# pyomo.SolverFactory("ipopt").solve(model_nlp)
# [model_nlp.y[0](), model_nlp.x[1](), model_nlp.x[1]()]

###############################################################################
# Функция, переводящая переменные решения pyomo в вектор
def DV_2_vec(model):
	x = [model.y[0], model.x[1], model.x[2]]
	return x

# нелинейная выпуклая функция цели
def obj(x):
 	return -(1000 - x[0]**2 - 2*x[1]**2 - x[2]**2 - x[0]*x[1] - x[0]*x[2])

# нелинейные ограничения - неравенства
def non_lin_cons(x):
	return [x[0]**2 + x[1]**2 + x[2]**2 - 25, x[1]**2 + x[2]**2 - 12]

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
# Нелинейная функция цели, есть нелинейные ограничения
###############################################################################
# с NLP
poa = mmaxgon_MINLP_POA(
	pyomo=pyomo,
	pyomo_MILP_model=model_milp,
	non_lin_obj_fun=obj,
	non_lin_constr_fun=non_lin_cons,
	decision_vars_to_vector_fun=DV_2_vec,
	eps=1e-6,
	milp_solver="cbc",
	#NLP_refiner_class=scipy_refiner_optimizer,
	NLP_projector_object=scipy_projector_optimizer_obj
)

start_time = time()
res1 = poa.solve(tolerance=1e-1, add_constr="ONE", tee=False)
print(time() - start_time)

start_time = time()
res2 = poa.solve(tolerance=1e-1, add_constr="ALL", tee=False)
print(time() - start_time)

# без NLP
poa = mmaxgon_MINLP_POA(
	pyomo=pyomo,
	pyomo_MILP_model=model_milp,
	non_lin_obj_fun=obj,
	non_lin_constr_fun=non_lin_cons,
	decision_vars_to_vector_fun=DV_2_vec,
	eps=1e-6,
	milp_solver="cbc"
)

start_time = time()
res3 = poa.solve(tolerance=1e-1, add_constr="ONE", tee=False)
print(time() - start_time)

start_time = time()
res4 = poa.solve(tolerance=1e-1, add_constr="ALL", tee=False)
print(time() - start_time)

print(res1)
print(res2)
print(res3)
print(res4)

###########################
# Все ограничения линейные, функция цели нелинейная
###########################
poa = mmaxgon_MINLP_POA(
	pyomo=pyomo,
	pyomo_MILP_model=model_milp,
	non_lin_obj_fun=obj,
	non_lin_constr_fun=None,
	decision_vars_to_vector_fun=DV_2_vec,
	eps=1e-6,
	milp_solver="cbc"
)
res5 = poa.solve(tolerance=1e-1, add_constr="ALL", tee=False)
print(res5)

###########################
# Есть нелинейные ограничения, функция цели - линейная
###########################
model_milp.obj = pyomo.Objective(expr= model_milp.y[0] + 2*model_milp.x[1] + model_milp.x[2])

poa = mmaxgon_MINLP_POA(
	pyomo=pyomo,
	pyomo_MILP_model=model_milp,
	non_lin_obj_fun=None,
	non_lin_constr_fun=non_lin_cons,
	decision_vars_to_vector_fun=DV_2_vec,
	eps=1e-6,
	milp_solver="cbc"
)
res6 = poa.solve(tolerance=1e-1, add_constr="ALL", tee=False)
print(res6)

###########################
# Функция цели и все ограничения линейные
###########################
poa = mmaxgon_MINLP_POA(
	pyomo=pyomo,
	pyomo_MILP_model=model_milp,
	non_lin_obj_fun=None,
	non_lin_constr_fun=None,
	decision_vars_to_vector_fun=DV_2_vec,
	eps=1e-6,
	milp_solver="cbc"
)
res7 = poa.solve(tolerance=1e-1, add_constr="ALL", tee=False)
print(res7)

model_milp.del_component(model_milp.obj)

###############################################################################
# Используется базовый MINLP-солвер
# Нелинейная функция цели, есть нелинейные ограничения
###############################################################################
model_nlp.obj = pyomo.Objective(expr = -(1000 - model_nlp.y[0]**2 - 2*model_nlp.x[1]**2 - model_nlp.x[2]**2 - model_nlp.y[0]*x[1] - model_nlp.y[0]*model_nlp.x[2]), sense=pyomo.minimize)

poa = mmaxgon_MINLP_POA(
	pyomo=pyomo,
	pyomo_MILP_model=model_nlp,
	non_lin_obj_fun=None,
	non_lin_constr_fun=None,
	decision_vars_to_vector_fun=DV_2_vec,
	eps=1e-6,
	milp_solver="couenne"
)

start_time = time()
res8 = poa.solve(tolerance=1e-1, add_constr="ONE", tee=False)
print(time() - start_time)

print(res8)
print(res3)

model_nlp.del_component(model_nlp.obj)
####################################################################
poa = mmaxgon_MINLP_POA(
	pyomo=pyomo,
	pyomo_MILP_model=model_nlp,
	non_lin_obj_fun=obj,
	non_lin_constr_fun=None,
	decision_vars_to_vector_fun=DV_2_vec,
	eps=1e-6,
	milp_solver="couenne"
)

start_time = time()
res9 = poa.solve(tolerance=1e-1, add_constr="ONE", tee=False)
print(time() - start_time)

print(res9)
print(res1)
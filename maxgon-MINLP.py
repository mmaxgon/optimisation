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
		pyomo_MILP_model,              # Модель MILP со всеми переменными решения и только линейными ограничениями
		non_lin_obj_fun,               # Нелинейная цункция цели
		non_lin_constr_fun,            # Нелинейные ограничения (list)
		decision_vars_to_vector_fun,   # Функция, комбинирующая переменные решения Pyomo в list
		eps=1e-6                       # Приращение аргумента для численного дифференцирования
	  ):
		self.__pyomo = pyomo
		self.__pyomo_MILP_model = pyomo_MILP_model
		self.__non_lin_obj_fun = non_lin_obj_fun
		self.__non_lin_constr_fun = non_lin_constr_fun
		self.__decision_vars_to_vector_fun = decision_vars_to_vector_fun
		self.__eps = eps

		if self.__non_lin_constr_fun == None:
			self.__non_lin_constr_fun = self.__constr_true

	# пустая функция ограничений
	def __constr_true(self, x):
		return [-1]

	# градиент для линейной аппроксимации цели
	def __get_linear_appr(self, fun, x):
		return opt.approx_fprime(x, fun, epsilon = self.__eps)

	# якобиан для линейной аппроксимации ограничений
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
		goal_best = np.Inf
		upper_bound = np.Inf
		lower_bound = -np.Inf
		feasible_x = []
		if_first_feasible = True
		iter_num = 0

		pyomo_MILP_model = copy.deepcopy(self.__pyomo_MILP_model)

		# Дополнительная переменная решений - верхняя граница цели (максимум от линейных аппроксимаций)
		pyomo_MILP_model.__mu = self.__pyomo.Var(domain=self.__pyomo.Reals, initialize=0)
		# На первом шаге накладываем на неё ограничение снизу, чтобы было решение
		pyomo_MILP_model.__mu_temp_cons = self.__pyomo.Constraint(expr=pyomo_MILP_model.__mu >= -1e6)

		# Нелинейные ограничения (пополняются для каждой итерации)
		pyomo_MILP_model.__non_lin_cons = self.__pyomo.ConstraintList()
		# ограничения на функцию цели (линейная аппроксимация после каждой успешной итерации)
		pyomo_MILP_model.__obj_cons = self.__pyomo.ConstraintList()

		# цель MILP
		pyomo_MILP_model.obj = self.__pyomo.Objective(expr=pyomo_MILP_model.__mu, sense=self.__pyomo.minimize)

		# солвер
		solver = self.__pyomo.SolverFactory('cbc')

		while True:
			iter_num += 1
			# MILP
			results = solver.solve(pyomo_MILP_model, tee = tee)

			if results.Solver()["Termination condition"] == self.__pyomo.TerminationCondition.infeasible:
				print("MILP не нашёл допустимого решения")
				return {
					"x": x_best,
					"obj": goal_best,
					"non_lin_constr_num": len(pyomo_MILP_model.__non_lin_cons),
					"goal_fun_constr_num": len(pyomo_MILP_model.__obj_cons),
					"iter_num": iter_num
				}

			# MILP-решение (даже недопустимое) даёт нижнюю границу
			lower_bound = self.__pyomo.value(pyomo_MILP_model.obj)

			# переводим переменные решения в вектор
			xvars = self.__decision_vars_to_vector_fun(pyomo_MILP_model)
			x = list(map(self.__pyomo.value, xvars))

			print(x)

			# получаем индексы для нарушенных нелинейных ограничений
			gx = self.__non_lin_constr_fun(x)
			ix_violated = list(np.where(np.array(gx) > 0)[0])

			# если решение допустимо
			if len(ix_violated) == 0:
				print("Feasible")
				# проверяем было ли уже данное решение (может быть долго)
				for y in feasible_x:
					if np.allclose(x, y):
						print("All close!")
						return {
							"x": x_best,
							"obj": goal_best,
							"non_lin_constr_num": len(pyomo_MILP_model.__non_lin_cons),
							"goal_fun_constr_num": len(pyomo_MILP_model.__obj_cons),
							"iter_num": iter_num
						}
				feasible_x.append(x)

				# если это первое допустимое решение - убираем ограничение на целевую переменную mu
				if if_first_feasible:
					pyomo_MILP_model.del_component(pyomo_MILP_model.__mu_temp_cons)
					if_first_feasible = False

				# добавляем новую аппроксимацию функции цели в ограничения
				fx = self.__non_lin_obj_fun(x)
				gradf = self.__get_linear_appr(self.__non_lin_obj_fun, x)
				xgradf = np.dot(x, gradf)
				pyomo_MILP_model.__obj_cons.add(
					fx - \
					xgradf + \
					sum(xvars[i] * gradf[i] for i in range(len(x))) <= pyomo_MILP_model.__mu
				)
				#pyomo_MILP_model.__obj_cons.pprint()

				# обновляем лучшее решение
				if fx < goal_best:
					x_best = copy.copy(x)
					goal_best = fx
					# верхняя граница - пока самое лучшее решение
					upper_bound = goal_best
					print(x_best, goal_best)

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
						"iter_num": iter_num
					}
			else:
				print("Not feasible")

				# добавляем линеаризацию нарушенных нелинейных ограничений
				# gx = self.__non_lin_constr_fun(x)
				# индексы нарушенных ограничений
				gx_violated = np.array(gx)[ix_violated]
				# если нарушенных ограничений несколько, а мы должны выбрать только одно
				if gx_violated.shape[0] > 1 and add_constr == "ONE":
					# оставляем только индекс с максимальным нарушением ограничения
					ix_most_violated = np.argmax(gx_violated)
					ix_violated = [ix_violated[ix_most_violated]]
					gx_violated = np.array(gx)[ix_violated]

				gradg_violated = self.__get_linear_appr_matrix(lambda x: np.array(self.__non_lin_constr_fun(x))[ix_violated], x)
				xgradg = np.matmul(gradg_violated, x)
				# добавляем новые аппроксимацию в ограничения
				for k in range(len(ix_violated)):
					pyomo_MILP_model.__non_lin_cons.add(
						gx_violated[k] - \
						xgradg[k] + \
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
# Задача как MILP
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

###############################################################################
# Функция, переводящая переменные решения в вектор
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
# Решаем
poa = mmaxgon_MINLP_POA(
	pyomo=pyomo,
	pyomo_MILP_model=model_milp,
	non_lin_obj_fun=obj,
	non_lin_constr_fun=non_lin_cons,
	decision_vars_to_vector_fun=DV_2_vec,
	eps=1e-6
)

start_time = time()
res1 = poa.solve(tolerance=1e-1, add_constr="ONE", tee=False)
print(time() - start_time)

start_time = time()
res2 = poa.solve(tolerance=1e-1, add_constr="ALL", tee=False)
print(time() - start_time)

print(res1)
print(res2)
###########################

poa = mmaxgon_MINLP_POA(
	pyomo=pyomo,
	pyomo_MILP_model=model_milp,
	non_lin_obj_fun=obj,
	non_lin_constr_fun=None,
	decision_vars_to_vector_fun=DV_2_vec,
	eps=1e-6
)
res3 = poa.solve(tolerance=1e-1, add_constr="ALL", tee=False)
print(res3)

non_lin_cons(res3["x"])
8 *res3["x"][0] + 14 * res3["x"][1] + 7 *res3["x"][2] - 56

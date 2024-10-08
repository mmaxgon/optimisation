"""
Выпуклая оболочка функции - максимальная выпуклая функция, не превышающая данную.
Если x0 - точка глобального минимума функции на множестве, то она является также (одним из) глобальных минимумов
выпуклой оболочки на этом множестве. Т.е. мы можем решать задачу выпуклой оптимизации с ФЦ в виде выпуклой оболочки,
чтобы найти глобальный минимум исходной функции.
Произвольная задача оптимизации -> выпуклая задача оптимизации.
"""
import numpy as np
from scipy.spatial import ConvexHull

# %matplotlib inline
import matplotlib.pyplot as plt


# проверка и графики
N = 1000
x = np.linspace(0., 1., N).reshape((N, 1))

f1 = np.sin(2. * np.pi * x) + np.cos(3. * np.pi * x)
f2 = np.cos(2. * np.pi * x) + np.sin(3. * np.pi * x)

func1 = lambda x_val: np.sin(2. * np.pi * x_val) + np.cos(3. * np.pi * x_val)
#################################################################################################################
# Выпуклая оболочка START
#################################################################################################################
# Выпуклая оболочка многогранника, образованного точками (x1, f(x1)), ..., (xN, f(xN))
def convex_envelope(x, f):
	"""
	x - матрица размерности (N, D), N - число точек, D - размерность пространства аргумента
	f - матрица (N, 1) из значений функции
	"""
	# Число точек
	N = x.shape[0]
	# Размерность пространства аргумента
	D = x.shape[1]

	# К массиву x добавляем две точки - в конце и начале списка, которые дублируют первую и последнюю точки
	x_pad = np.empty((N + 2, D))
	x_pad[0, :] = x[0, :]
	x_pad[-1, :] = x[-1, :]
	x_pad[1:-1, :] = x

	# К массиву значений функции добавляем две точки - в начале и конце, значения которых - максимальные значения функции + 1
	f_pad = np.empty((N + 2, 1))
	f_pad[(0, -1), 0] = np.max(f) + 1.
	f_pad[1:-1, 0] = f[:, 0]

	# график функции
	epi = np.column_stack((x_pad, f_pad[:, 0]))
	# выпуклая оболочка графика функции
	hull = ConvexHull(epi)
	# # индексы вершин, образующих опорные точки выпуклой оболочки графика функции (без первой и последней точки)
	# result = [v - 1 for v in hull.vertices if 0 < v <= N]
	# result.sort()
	# result = np.array(result)
	# return result
	return hull

def get_convex_envelope_value(x_val, hull):
	"""
	Args:
		x_val: вектор аргумента
		hull: выпуклая оболочка многогранника, образованного точками (x1, f(x1)), ..., (xN, f(xN))
		Это выпуклая оболочка аппроксимации эпиграфа f по выборке
	Returns:
		значение функции-аппроксимации выпуклой оболочки
	"""
	# Ax + by <= c - уравнение полуплоскости
	A = hull.equations[:, 0:-2]
	b = hull.equations[:, -2]
	c = hull.equations[:, -1]
	ix = b < 0
	A = A[ix, :]
	b = b[ix]
	c = -c[ix]
	y = (c - A @ x_val) / b
	result = np.max(y)
	return result

#################################################################################################################
# Выпуклая оболочка END
#################################################################################################################

plt.figure(figsize=(10, 5))
plt.subplot(211)
plt.title('Выпуклые оболочки функций')
plt.plot(x, f1, label='Функция')
# result_1 = convex_envelope(x, f1)
# plt.plot(x[result_1, :], f1[result_1], label='Выпуклая оболочка')
hull_1 = convex_envelope(x, f1)
y_1 = [get_convex_envelope_value(x_val, hull_1) for x_val in x]
plt.plot(x[:, 0], y_1, label='Выпуклая оболочка')
plt.legend()
plt.subplot(212)
plt.plot(x, f2, label='Функция')
hull_2 = convex_envelope(x, f2)
y_2 = [get_convex_envelope_value(x_val, hull_2) for x_val in x]
plt.plot(x[:, 0], y_2, label='Выпуклая оболочка')
plt.legend()
plt.show()

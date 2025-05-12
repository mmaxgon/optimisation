# Dual Lower Bound
# Более эффективная нижняя граница оптимального решения для линейной бинарной задачи
import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import milp
import torch
import matplotlib.pyplot as plt

# число переменных
N = 100
# число ограничений
M = 40
np.random.seed(12)
# Вектор ф-ции цели
c = np.array([np.random.uniform() for j in range(N)])
c.shape
# Матрица ограничений
A = np.array([[np.random.uniform() for j in range(N)] for i in range(M)])
A.shape
# Столбцы ограничений
lb = np.zeros((M,))
ub = np.ones((M,))

# Определяем линейные ограничения: lb <= Ax <= ub
constraint = LinearConstraint(A=A, lb=lb, ub=ub)

# Определяем границы переменных: от 0 до 1
bounds = Bounds(lb=np.zeros((N,)), ub=np.ones((N,)))

# Решаем релаксированную задачу
res = milp(c=-c, bounds=bounds, constraints=constraint)
relaxed_x = res.x
relaxed_LB = res.fun
print(relaxed_LB)

# Решаем исходную задачу
res = milp(c=-c, bounds=bounds, constraints=constraint, integrality=[1]*N)
result_x = res.x
result = res.fun
print(result)

# Получаем нижнюю границу Dual Lower Bound
A = torch.tensor(A)
b = torch.tensor(ub)
c = torch.tensor(c)

def get_loss():
	Aty = torch.matmul(torch.t(A), y)
	yb = torch.matmul(y, torch.t(b))
	loss = yb
	for j in range(N):
		loss += torch.nn.functional.relu(c[j] - Aty[j])
	return loss

y = torch.zeros((M,), requires_grad=True, dtype=torch.double)
opt = torch.optim.Adam([y], lr=0.01)
min_loss = get_loss().item()
for k in range(int(5e3)):
	opt.zero_grad()
	loss = get_loss()
	loss.backward()
	opt.step()
	# ограничение на неотрицательные множители Лагранжа y
	with torch.no_grad():
		y[:] = torch.clamp(y, min=0)
	print(loss.item())
	if min_loss > loss.item():
		min_loss = loss.item()
print(min_loss)
dual_lower_bound = -min_loss

# Результаты
plt.plot([relaxed_LB-0.1, result+0.1],[0, 0])
plt.plot([relaxed_LB], [0], "o", label=f"relaxed bound {round(relaxed_LB, 5)}", color="red")
plt.plot([dual_lower_bound], [0], "o", label=f"dual bound {round(dual_lower_bound, 5)}", color="blue")
plt.plot([result], [0], "o", label=f"actual {round(result, 5)}", color="green")
plt.legend()
plt.savefig('DLB.png')
plt.close()
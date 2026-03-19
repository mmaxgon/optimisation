import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from time import time
import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
from scipy.optimize import milp
import pandas as pd
import pyscipopt as scip
import matplotlib.pyplot as plt

# число переменных
N = 150
# число ограничений
M = 60
np.random.seed(123)
# Вектор ф-ции цели
c = np.array([np.random.randint(1, 10) for j in range(N)])
c.shape
# Матрица ограничений
A = np.array([[np.random.randint(0, 10) for j in range(N)] for i in range(M)])
A.shape
# Столбцы ограничений
lb = np.zeros((M,))
ub = 100 * np.ones((M,))

# Определяем линейные ограничения: lb <= Ax <= ub
constraint = LinearConstraint(A=A, lb=lb, ub=ub)

# Определяем границы переменных: от 0 до 1
bounds = Bounds(lb=np.zeros((N,)), ub=np.ones((N,)))

# Решаем исходную задачу
def solve_milp():
    res = milp(c=-c, bounds=bounds, constraints=constraint, integrality=[1]*N)
    result_x = res.x
    result = res.fun
    return (result_x, -result)

# Решаем релаксированную задачу
def solve_lp():
    res = milp(c=-c, bounds=bounds, constraints=constraint)
    result_x = res.x
    result = res.fun
    return (result_x, -result)

####################################################################
# Решаем штрафами
####################################################################

def is_constraints_ok(x, constraint, bounds):
    res = np.empty(shape=(0), dtype=np.bool)
    res = np.concatenate((res, x <= bounds.ub))
    res = np.concatenate((res, x >= bounds.lb))
    res = np.concatenate((res, constraint.A @ x <= constraint.ub))
    res = np.concatenate((res, constraint.A @ x >= constraint.lb))
    return bool(np.all(res))

def binary_penalty(x, gamma=1.0):
    """Штраф за отклонение от бинарности: gamma * sum(x_i * (1-x_i))"""
    return gamma * np.sum(x * (1 - x))

def binary_penalty_gradient(x, gamma=1.0):
    """Градиент штрафа: gamma * (1 - 2*x_i) для каждой компоненты"""
    return gamma * (1 - 2*x)

def total_objective(x, f_obj, gamma=1.0, eps=1e-6):
    """Исходная целевая функция + барьер"""
    return f_obj(x) + binary_penalty(x, gamma)

def total_gradient(x, grad_obj, gamma=1.0, eps=1e-6):
    """Градиент исходной функции + барьера"""
    return grad_obj(x) + binary_penalty_gradient(x, gamma)

def obj(x):
    return -c @ x

def grad_obj(x):
    return -c

def solve_binary_penalty(
    x0, # Начальная точка
    gammas, # Решаем с разными штрафами
):
    solutions = []

    for gamma in gammas:
        res = minimize(
            fun=lambda x: total_objective(x, obj, gamma),
            bounds=bounds,
            constraints=[constraint],
            x0=x0,
            method='SLSQP', #'trust-constr',
            jac=lambda x: total_gradient(x, grad_obj, gamma),
        )
        x0 = res.x
        x1 = np.round(x0)
        # Возвращаем первое допустимое целочисленное решение
        if is_constraints_ok(x1, constraint=constraint, bounds=bounds):
            solutions.append((x1, -obj(x1)))
            return  solutions

        solutions.append((res.x, -res.fun))
    return solutions

########################################################################

start = time()
(result_x_milp, result_milp) = solve_milp()
duration_milp = time() - start

start = time()
(result_x_lp, result_lp) = solve_lp()
duration_lp = time() - start

start = time()
solutions = solve_binary_penalty(x0=result_x_lp, gammas=[1e2, 1e3, 1e4, 1e5, 1e6])
(result_x_binary_penalty, result_binary_penalty) = solutions[len(solutions)-1]
duration_binary_penalty = time() - start

print(f"MILP: {duration_milp}, \n {result_milp}, \n {result_x_milp}")
print(f"LP: {duration_lp}, \n {result_lp}, \n {result_x_lp}")
print(f"PENALTY: {duration_binary_penalty}, \n {result_x_binary_penalty}, \n {result_binary_penalty}")

df_results = pd.DataFrame({
    "Algorithm": ["LP", "MILP", "PENALTY"],
    "Duration": [duration_lp, duration_milp, duration_binary_penalty],
    "Goal": [result_lp, result_milp, result_binary_penalty]
})

print(df_results)

#############################################################
# SCIP
#############################################################
def solve_scip(initial_sol=None):
    model = scip.Model()

    x = [model.addVar(name=f"x[{j}]", vtype="B") for j in range(N)]

    for i in range(M):
        model.addCons(cons=scip.quicksum(A[i, j] * x[j] for j in range(N)) <= ub[i])

    model.setObjective(expr=scip.quicksum(c[j] * x[j] for j in range(N)), sense='maximize')

    if initial_sol is not None:
        sol_1 = model.createSol()
        for j in range(N):
            model.setSolVal(sol_1, x[j], initial_sol[j])
        # Проверяем и добавляем решение
        accepted = model.addSol(sol_1, free=False)
        print(f"Accepted: {accepted}")

    model.setParam("display/verblevel", 2)
    # model.setParam("limits/gap", 2)
    model.setParam("limits/absgap", 8)

    model.optimize()
    sol = model.getBestSol()

    if model.getStatus() != "infeasible":
        result_scip = model.getObjVal()
        result_x_scip = np.array([sol[x[j]] for j in range(N)])
        return (result_x_scip, result_scip)
    else:
        print("No solution found")
        return None

start = time()
(result_x_scip, result_scip) = solve_scip()
duration_scip = time() - start

df_results = pd.concat((df_results, pd.DataFrame([["SCIP", duration_scip, result_scip]], columns=df_results.columns)))

start = time()
(result_x_scip_ws, result_scip_ws) = solve_scip(initial_sol=result_x_binary_penalty)
duration_scip_ws = time() - start

df_results = pd.concat((df_results, pd.DataFrame([["SCIP WS", duration_scip_ws, result_scip_ws]], columns=df_results.columns)))

print(df_results)
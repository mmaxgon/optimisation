# pip install pyscipopt
# https://github.com/scipopt/PySCIPOpt
# https://scipopt.github.io/PySCIPOpt/docs/html/classpyscipopt_1_1scip_1_1Model.html
# custom constraints: https://github.com/scipopt/PySCIPOpt/blob/master/tests/test_conshdlr.py
# parameters: https://www.scipopt.org/doc/html/PARAMETERS.php

import numpy as np
from pyscipopt import Model as scip_Model

#########################################################################################
scip_model = scip_Model("SCIP Example")
# scip_model.enableReoptimization()

# переменные решения
# vtype: type of the variable: 'C' continuous, 'I' integer, 'B' binary, and 'M' implicit integer
scip_model_y = [scip_model.addVar(name="y", vtype="C", lb=0.0, ub=10.0)]
scip_model_x = [scip_model.addVar(name="x{0}".format(i), vtype="I", lb=0, ub=10) for i in range(2)]

# линейные ограничения
scip_model_lin_cons = scip_model.addCons(
	cons=8 * scip_model_y[0] + 14 * scip_model_x[0] + 7 * scip_model_x[1] - 56 == 0,
	name="lin_constr"
)
# нелинейные ограничения
scip_model_non_lin_cons1 = scip_model.addCons(
	cons=scip_model_y[0]**2 + scip_model_x[0]**2 + scip_model_x[1]**2 <= 25,
	name="nonlin_constr_1"
)
scip_model_non_lin_cons2 = scip_model.addCons(
	cons=scip_model_x[0]**2 + scip_model_x[1]**2 <= 12,
	name="nonlin_constr_2"
)

# функция цели ! Nonlinear objective functions are not supported !
# Через временное ограничение
scip_model_mu = scip_model.addVar(name="mu", vtype="C", lb=-1e9)

scip_model_obj_cons = scip_model.addCons(
	cons=-(1000 - scip_model_y[0]**2 - 2*scip_model_x[0]**2 - scip_model_x[1]**2 - scip_model_y[0]*scip_model_x[0] - scip_model_y[0]*scip_model_x[1]) <= scip_model_mu,
	name="objective_constr"
)
scip_model.setObjective(scip_model_mu, sense='minimize')
# задана ли функция цели
print(str(scip_model.getObjective()) != "Expr({})" and str(scip_model.getObjective()) != "")

scip_model.setParam('limits/time', 600)
scip_model.setParam('constraints/setppc/cliquelifting', True)

# решаем
scip_model.optimize()
sol = scip_model.getBestSol()

# решение
if scip_model.getStatus() != "infeasible":
	print(scip_model.getObjVal())
	print([sol[scip_model_y[0]], sol[scip_model_x[0]], sol[scip_model_x[1]]])
else:
	print("No solution found")

print([(x, sol[x]) for x in scip_model.getVars()])

scip_model.getStage()
# del sol
scip_model.freeTransform()
scip_model.getStage()
# scip_model.freeReoptSolve()
# добавляем временное ограничение
scip_model_mu_temp = scip_model.addCons(cons=scip_model_mu >= -970, name="temp", removable="True")
scip_model.optimize()
sol = scip_model.getBestSol()
print(scip_model.getObjVal())
print([sol[scip_model_y[0]], sol[scip_model_x[0]], sol[scip_model_x[1]]])

# удаление ограничений
# del sol
scip_model.freeTransform()
# scip_model.freeReoptSolve()
scip_model.delCons(scip_model_mu_temp)
scip_model.optimize()
sol = scip_model.getBestSol()
print(scip_model.getObjVal())
print([sol[scip_model_y[0]], sol[scip_model_x[0]], sol[scip_model_x[1]]])

# двойственные переменные: только для линейных ограничений
# 'dual solution values not available for constraints of type ', 'nonlinear'
[scip_model.getDualsolLinear(c) for c in scip_model.getConss()]





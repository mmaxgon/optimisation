import numpy as np
import pyscipopt as scip

eps = 1e-4
# Длина доски
L = 15
# Длины брусков
l_lens = [1, 3, 5, 7]
# Число разных брусков
l_count = len(l_lens)
# Потребности в брусках
l_demands = np.array([19, 9, 7, 5])

# Первоначальный набор шаблонов нарезки (тупой: шаблон состоит из одного бруска)
patterns = np.eye(l_count, dtype=int)

"""
Решаем мастер-задачу: 
сколько каждого шаблона из текущего набора надо использовать, чтобы 
- удовлетворить всем потребностям и 
- уменьшить суммарное число используемых досок
"""

def get_master_model(patterns, final=False):
    model = scip.Model("Master Problem")

    dummy = model.addVar(name="dummy", lb=0, ub=0, obj=0)

    # сколько на данный момент паттернов
    m = patterns.shape[1]

    # x[j] - сколько паттернов типа j участвуют в раскройке
    x = {}
    for j in range(m):
        vec = patterns[:,j]
        x[j] = model.addVar(name=f"x_{j}", vtype="I", lb=0)

    for i in range(l_count):
        # должны обеспечить потребность в каждой детали
        model.addCons(
            dummy + scip.quicksum(patterns[i, j] * x[j] for j in range(m)) >= l_demands[i]
        )

    # Минимизируем число используемых паттернов (брусков)
    model.setObjective(
        expr=scip.quicksum(x[j] for j in range(m)),
        sense="minimize"
    )

    if not final:
        # Чтобы решение не нашлось эвристикой, отключаем presolve и прочие манипуляции - нам нужны двойственные переменные
        model.setPresolve(scip.SCIP_PARAMSETTING.OFF)
        model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)
        model.disablePropagation()
    model.setParam("display/verblevel", 0)

    return model

def get_and_solve_sub_model(duals):
    # Новый паттерн раскройки
    sub_model = scip.Model("Sub Problem")
    # Сколько брусков каждой длины входит в паттерн
    y = {}
    for i in range(l_count):
        y[i] = sub_model.addVar(name=f"y_{i}", lb=0, ub=L // l_lens[i], vtype="I")
    # Бруски должны помещаться в длину
    sub_model.addCons(scip.quicksum(l_lens[i] * y[i] for i in range(l_count)) <= L)
    # Минимизируем приведённые цены: 1 - duals * y
    sub_model.setObjective(scip.quicksum(y[i] * duals[i] for i in range(l_count)), sense="maximize")
    sub_model.setParam("display/verblevel", 0)
    sub_model.optimize()
    status = sub_model.getStatus()
    if status == "optimal":
        # model.printStatistics()
        sol = sub_model.getBestSol()
        obj = sub_model.getObjVal(sol)
        if obj < 1 + eps:
            return None
        new_pattern = [sol[y[i]] for i in range(l_count)]
        return new_pattern
    else:
        return None

def do_one_step(patterns):
    model = get_master_model(patterns)
    model.relax()
    model.optimize()
    status = model.getStatus()

    if status == "optimal":
        sol = model.getBestSol()
        obj = model.getObjVal(sol)
        # model.printStatistics()
        duals = [model.getDualsolLinear(cons) for cons in model.getConss()]
        new_pattern = get_and_solve_sub_model(duals)
        if new_pattern is not None:
            patterns = np.column_stack((patterns, np.array(new_pattern, dtype=int).reshape(-1, 1)))
            return (True, patterns, obj)
        else:
            return (False, patterns, obj)
    else:
        raise ValueError("No solution found")

def do_last_step(patterns):
    model = get_master_model(patterns, final=True)
    model.optimize()
    status = model.getStatus()

    if status == "optimal":
        sol = model.getBestSol()
        obj = model.getObjVal(sol)
        # model.printStatistics()
        res = {var.name: sol[var] for var in model.getVars() if sol[var] >= 1}
        return (res, obj)
    else:
        raise ValueError("No solution found")

go = True
while go:
    (go, patterns, rel_obj) = do_one_step(patterns)
    print(patterns)

print(f"ЦФ релаксированная: {rel_obj}")
(res, obj) = do_last_step(patterns)
print(f"ЦФ целочисленная: {obj}")
print(res)


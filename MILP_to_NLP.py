from pyscipopt import Model, quicksum
import numpy as np

def solve_primal_penalty_method(A, b, c, max_iterations=20, initial_rho=0.1, rho_multiplier=5.0, tolerance=1e-6):
    """
    Решает прямую задачу методом последовательного квадратичного штрафа.
    Возвращает верхнюю границу (UB) и приближённое решение.
    """
    m, n = A.shape  # m ограничений, n переменных
    
    # Создаём модель и переменные ОДИН РАЗ
    model = Model("PrimalPenaltyMethod")
    x_vars = []
    for i in range(n):
        x_var = model.addVar(f"x_{i}", lb=0.0, ub=1.0, vtype="C")
        x_vars.append(x_var)
    
    # Добавляем ограничения Ax <= b
    for j in range(m):
        constraint_expr = quicksum(A[j, i] * x_vars[i] for i in range(n))
        model.addCons(constraint_expr <= b[j])
    
    # Предварительно вычисляем линейную часть целевой функции
    linear_objective = quicksum(c[i] * x_vars[i] for i in range(n))
    
    # Инициализация
    current_rho = initial_rho
    current_solution = np.full(n, 0.5)
    best_ub = float('inf')
    best_solution = None
    
    print("=== PRIMAL PENALTY METHOD (Upper Bound) ===")
    print(f"{'Iter':>4} {'Rho':>8} {'Objective':>12} {'Max Gap':>10} {'Best UB':>12}")
    print("-" * 60)
    
    for iteration in range(max_iterations + 1):
        # Формируем целевую функцию для текущего rho
        if current_rho == 0:
            objective_expr = linear_objective
        else:
            quadratic_penalty = current_rho * quicksum(x_vars[i] - x_vars[i]*x_vars[i] for i in range(n))
            objective_expr = linear_objective + quadratic_penalty
        
        model.setObjective(objective_expr, "minimize")
        
        # Добавляем начальное решение (кроме первой итерации)
        if iteration > 0:
            sol = model.createSol()
            for i in range(n):
                model.setSolVal(sol, x_vars[i], current_solution[i])
            model.addSol(sol, free=True)
        
        # Решаем задачу
        model.optimize()
        
        # Извлекаем решение
        current_solution = np.array([model.getVal(x_var) for x_var in x_vars])
        current_obj = model.getObjVal()
        
        # Анализируем бинарность и обновляем лучшую верхнюю границу
        binary_gaps = np.minimum(current_solution, 1 - current_solution)
        max_binary_gap = np.max(binary_gaps)
        
        # Округляем решение для получения допустимой верхней границы
        rounded_solution = np.round(current_solution)
        rounded_obj = np.dot(c, rounded_solution)
        
        if rounded_obj < best_ub:
            best_ub = rounded_obj
            best_solution = rounded_solution.copy()
        
        print(f"{iteration:4d} {current_rho:8.3f} {current_obj:12.6f} {max_binary_gap:10.6f} {best_ub:12.6f}")
        
        # Критерий остановки
        if max_binary_gap < tolerance:
            print(f"Converged to binary solution after {iteration} iterations!")
            break
        
        # Подготовка к следующей итерации
        if iteration == 0:
            current_rho = initial_rho
        else:
            current_rho *= rho_multiplier
    
    return best_ub, best_solution

def solve_dual_lagrangian_problem(A, b, c, max_iterations=1000):
    """
    Решает двойственную лагранжеву задачу для получения нижней границы.
    Возвращает нижнюю границу (LB) и оптимальные множители Лагранжа.
    """
    m, n = A.shape  # m ограничений, n переменных
    
    model = Model("LagrangianDual")
    
    # Создаём переменные для множителей Лагранжа (λ >= 0)
    lambda_vars = []
    for j in range(m):
        l_var = model.addVar(f"lambda_{j}", lb=0.0, ub=None, vtype="C")
        lambda_vars.append(l_var)
    
    # Явное задание двойственной функции: max [ -bᵀλ + Σ min(0, c_i + a_iᵀλ) ]
    
    # Первая часть: -bᵀλ
    linear_part = -quicksum(b[j] * lambda_vars[j] for j in range(m))
    
    # Вторая часть: Σ min(0, c_i + a_iᵀλ)
    min_sum = 0
    for i in range(n):
        # Вычисляем c_i + a_iᵀλ = c_i + Σ A[j,i] * λ_j
        inner_product = c[i]
        for j in range(m):
            inner_product += A[j, i] * lambda_vars[j]
        
        # Создаём вспомогательную переменную для min(0, inner_product)
        aux_var = model.addVar(f"min_{i}", lb=None, ub=0.0, vtype="C")
        
        # Добавляем ограничения: aux_var <= 0, aux_var <= inner_product
        model.addCons(aux_var <= 0)
        model.addCons(aux_var <= inner_product)
        
        min_sum += aux_var
    
    # Устанавливаем целевую функцию
    model.setObjective(linear_part + min_sum, "maximize")
    
    # Решаем задачу
    model.optimize()
    
    # Извлекаем результаты
    lb_value = model.getObjVal()
    lambda_values = np.array([model.getVal(lambda_var) for lambda_var in lambda_vars])
    
    print(f"\n=== LAGRANGIAN DUAL (Lower Bound) ===")
    print(f"Lower bound (LB): {lb_value:.6f}")
    
    return lb_value, lambda_values

def check_feasibility(A, b, x_solution):
    """Проверяет выполнение ограничений Ax <= b для решения x."""
    residuals = A.dot(x_solution) - b
    max_violation = np.max(residuals)
    is_feasible = max_violation <= 1e-6
    return is_feasible, max_violation, residuals

# Пример использования
if __name__ == "__main__":
    # Пример данных: небольшая тестовая задача
    # min cᵀx, subject to Ax <= b, x ∈ {0,1}
    np.random.seed(42)
    
    n = 10  # количество переменных
    m = 5   # количество ограничений
    
    # Генерируем случайные данные
    c = np.random.randn(n)  # Целевая функция
    A = np.random.randn(m, n)  # Матрица ограничений
    b = np.random.rand(m) * 2 + 1  # Правая часть ограничений
    
    print("Problem dimensions:", A.shape)
    print("Objective coefficients:", c)
    print("Right-hand side:", b)
    
    # 1. Решаем прямую задачу (получаем верхнюю границу UB)
    ub, x_solution = solve_primal_penalty_method(A, b, c)
    
    # 2. Решаем двойственную задачу (получаем нижнюю границу LB)  
    lb, lambda_values = solve_dual_lagrangian_problem(A, b, c)
    
    # 3. Анализ результатов
    print(f"\n=== FINAL RESULTS ===")
    print(f"Upper bound (UB): {ub:.6f}")
    print(f"Lower bound (LB): {lb:.6f}")
    
    if lb > -np.inf:
        optimality_gap = (ub - lb) / max(abs(lb), 1e-10) * 100
        print(f"Optimality gap: {optimality_gap:.2f}%")
    else:
        print("Lower bound is -inf, cannot compute gap")
    
    # Проверяем допустимость решения
    is_feasible, max_violation, _ = check_feasibility(A, b, x_solution)
    print(f"Solution feasible: {is_feasible} (max violation: {max_violation:.6f})")
    print(f"Solution: {x_solution}")
    
    # Дополнительная информация
    print(f"\nLagrange multipliers: {lambda_values}")
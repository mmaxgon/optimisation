"""
python -m pip install --upgrade scipy
scipy.__version__ == 1.10.1
"""
import numpy as np
from scipy.sparse import bsr_array, lil_array
from scipy.optimize import LinearConstraint
from scipy.optimize import milp

# c = -np.array([0, 1])
c = lil_array((1, 2))
c[0, 1] = -1
print(c.toarray())
# A = np.array([[-1, 1], [3, 2], [2, 3]])
A = bsr_array(([-1, 1, 3, 2, 2, 3], ([0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1])), shape=(3, 2))
# print(A.toarray())
b_u = np.array([1, 12, 12])
b_l = np.full_like(b_u, -np.inf)

# признак целочисленности
integrality = np.ones_like(c)

constraints = LinearConstraint(A, b_l, b_u)

res = milp(c=c.toarray().ravel(), constraints=constraints, integrality=np.zeros_like(c))
print(res.x)
res = milp(c=c.toarray().ravel(), constraints=constraints, integrality=integrality)
print(res.x)


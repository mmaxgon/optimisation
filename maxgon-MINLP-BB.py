"""A Mixed-Integer solver based on scipy.optimize.linprog. 

This code implements branch-and-bound on the linear relaxation of a given
mixed-integer program. It requires numpy and scipy.optimize.


Usage examples are given in the test() and test2() functions. Parameters of
MipModel are mostly as documented in scipy.optimize.linprog. The additional
parameter "int_vars" gives a sequence of indexes of the variables that should be
constrained to integer solutions.

Typical usage will follow the pattern of:

    import mip
    mip_model = mip.MipModel(C, minimize, Aub, bub, Aeq, beq, bounds, int_vars)
    mip.set_debug_prints(True)
    best_solution = mip.find_solutions(mip_model, depth_limit=C.shape[0] - 1)
    print(f'  soln: {best_solution.x}\n  objective value: {best_solution.fun}\n'
         f'success: '{best_solution.success}')

"""
# import collections
from dataclasses import dataclass, field, replace
import datetime
import itertools
import numpy as np
from scipy.optimize import linprog
import sys
from typing import Generator, List, MutableSequence, Optional, Sequence, Tuple

_DEBUG = True

def set_debug_prints(is_on):
	global _DEBUG
	_DEBUG = is_on

@dataclass(frozen=True)
class MipResult:
	model: 'MipModel'
	fun: float = np.inf  # !!! поменять на obj
	x: np.ndarray = np.array([])
	success: bool = False

	# соблюдается ли целочисленность дискретных переменных
	def is_int_soln(self) -> bool:
		return np.all(self.x[self.model.int_vars] == np.floor(self.x[self.model.int_vars]))

	# индексы целочисленных переменных, значения в которых не целые числа (после успешного решения)
	def vars_to_split(self) -> List[int]:
		if self.success:
			return list(np.where(self.x[self.model.int_vars] != np.floor(self.x[self.model.int_vars]))[0])
		else:
			return []

@dataclass(frozen=True)
class MipModel:
	c: np.ndarray
	minimize: bool = True
	Aub: Optional[np.ndarray] = None
	bub: Optional[np.ndarray] = None
	Aeq: Optional[np.ndarray] = None
	beq: Optional[np.ndarray] = None
	# границы переменных в формате ((l1,u1), (l2,u2), (l3,u3))
	bounds: Tuple = ()
	int_vars: Sequence[int] = field(default_factory=list)
	method: str = 'revised simplex'
	# границы переменных в формате np.array([l1,l2,l3], [u1,u2,u3])
	clip_bound: np.ndarray = field(init=False)
	
	def __post_init__(self):
		if not self.minimize:
			object.__setattr__(self, 'c', -1 * self.c)
		if not self.bounds:
			# размножение границ на [[0,0,...], [inf, inf,...]] по числу переменных
			object.__setattr__(self, 'clip_bound', np.tile(np.array([[0], [np.inf]]), self.c.shape[0]))
		else:
			lower, upper = zip(*self.bounds)
			# для upper переводим None в inf
			numeric_upper = [np.inf if u is None else u for u in upper]
			# формируем границы в формате np.array([l1,l2,l3], [u1,u2,u3])
			object.__setattr__(self, 'clip_bound', np.array((lower, numeric_upper), dtype=np.float64))

	def solve(self) -> MipResult:
		res = linprog(
			self.c,
			A_ub=self.Aub,
			b_ub=self.bub,
			A_eq=self.Aeq,
		    b_eq=self.beq,
			bounds=self.bounds,
			method=self.method
		)
		if res["success"]:
			# получаем новое решение
			result = MipResult(
				self,
				(int(self.minimize) * 2 - 1) * res['fun'], # умножаем результат на -1 если задача максимизации
				res['x'].clip(self.clip_bound[0], self.clip_bound[1]), # приводим решение в границы
				res['success'])
		else:
			# получаем пустое решение
			result = MipResult(self)
		return result

	# передаём дальше 2 модели, разбитые по переменной с индексом var_i по значению value или посередине(?)
	def split_on_var(
		self,
		var_i: int,
		value: Optional[float] = None
	) -> Generator['MipModel', None, None]:
		# индекс var_i в рамках?
		assert var_i in range(len(self.bounds)), 'Bad variable index for split'
		# границы на переменную var_i
		bound_i = self.bounds[var_i]
		if value is None: # не передано значение разбиения
			if bound_i[1] is not None: # верхняя граница переменной задана
				# определяем значение разбиения как среднее арифметичекое между верхней и нижней границей
				value = self.clip_bound[:, var_i].sum() / 2.0
			else: # верхняя граница не задана
				# возвращаем себя
				yield self
				return
		# We know where to split, have to treat None carefully.
		elif bound_i[1] is None: # передано значение разбиения, не задана верхняя граница
			bound_i = (bound_i[0], np.inf) # верхняя граница - inf
		# теперь у нас есть и значение разбиения и верхняя граница
		assert(value >= bound_i[0] and value <= bound_i[1], 'Bad value in split.')
		# новые границы слева
		new_bounds = (
			*self.bounds[:var_i], # границы на предыдущие переменные
			(bound_i[0], max(bound_i[0], np.floor(value))),
			*self.bounds[var_i + 1:] # границы на последующие переменные
		)
		# задача слева: копируем себя с изменёнными границами
		yield replace(self, bounds=new_bounds)
		# новая граница справа
		if np.isinf(bound_i[1]):
			new_upper = (np.ceil(value), None)
		else:
			new_upper = (min(np.ceil(value), bound_i[1]), bound_i[1])
		new_bounds = (
			*self.bounds[:var_i],
			new_upper,
			*self.bounds[var_i + 1:]
		)
		# задача справа: копируем себя с изменёнными границами
		yield replace(self, bounds=new_bounds)

def filter_result(
	result: MipResult,
	best_soln: MipResult,
    results: MutableSequence[MipResult] = []
) -> Tuple[MutableSequence[MipResult], MipResult]:
	if result.success:
		if result.is_int_soln() and result.fun <= best_soln.fun:
			if _DEBUG:
				print('\n', f'  {result.x}: {result.fun}')
				sys.stdout.flush()
			best_soln = result
		else:
			results.append(result)
	return results, best_soln

def walk_branches(solutions: Sequence[MipResult], best_soln: MipResult,
                  depth_limit: int) -> Tuple[MipResult, int]:
	"""Does depth-first search w/ branch & bound on the models in `solutions`.
 
	This function keeps track of a best solution and branches on variables
	for each solution in `solutions` up to a depth of `depth_limit`. Intermediate
	best solutions are printed with objective function and timestamp.
 
	Args:
	  solutions: Iterable of MipResult, assumes "not result.is_int_soln()"
	  best_soln: MipResult of the best integer solution (by fun) so far.
	  depth_limit: Integer depth limit of the search. This function will use up
		 to (2**depth_limit + 1) * 8 bytes of memory to store hashes that
		 identify previoulsy solved instances. Shallow copies are used
		 aggressively, but memory may be an issue as a function of this
		 parameter. CPU time _certainly_ is. Remove use of "repeats" for constant
		 memory utilization.
 
	Returns: The best MipResult obtained in the search (by result.fun) and a
		  count of stopped branches.
	"""
	
	def chirp(top_count):
		if _DEBUG:
			now = datetime.datetime.now()
			print(f'{len(stack):3} {now.strftime("%m-%d %H:%M:%S")}')
			sys.stdout.flush()
	
	num_cut = 0
	repeats = set()
	# Put solutions in a stack paired with variable indexs that need splitting.
	# Pop an element, split it, and push any new branches back on, reducing vars
	# to split by one element. Repeat until stack_limit reached or stack is
	# empty.
	stack = [(s, s.vars_to_split()) for s in solutions]
	stack_limit = len(stack) + depth_limit + 1
	start_depth = len(stack)
	chirp(start_depth)
	while stack and (len(stack) <= stack_limit):
		if len(stack) < start_depth:
			chirp(len(stack))
			start_depth = len(stack)
		cur_depth = len(stack) - start_depth
		soln, split_vars = stack.pop()
		var_i = split_vars.pop()
		if split_vars:
			stack.append((soln, split_vars))
		for model in soln.model.split_on_var(var_i, soln.x[var_i]):
			# Skip any set of bounds we've (probably) already seen.
			bounds_hash = hash(model.bounds)
			if bounds_hash in repeats:
				continue
			# Filter out any infeasible or integer solutions to avoid further
			# processing.
			result = model.solve()
			if result.success:
				if result.is_int_soln() and result.fun <= best_soln.fun:
					if _DEBUG:
						now = datetime.datetime.now()
						time = now.strftime("%H:%M:%S")
						print(f'\n  {result.x}: {result.fun} (d={cur_depth}, {time})')
						sys.stdout.flush()
					best_soln = result
				elif len(stack) < stack_limit:
					new_vars = result.vars_to_split()
					if new_vars:
						stack.append((result, new_vars))
				else:
					num_cut += 1
			# Save bounds for all but the last layer (which uses too much storage
			# for the compute it saves.)
			if cur_depth < depth_limit - 1:
				repeats.add(bounds_hash)
	return best_soln, num_cut


def find_solutions(mip_model: MipModel, depth_limit: int = 0) -> MipResult:
	"""Searches for mixed integer solutions to mip_model with branch & bound.
 
	This function searches for mixed-integer solutions to the given model by
	branching on the linear relaxtion of the model. Results will be returned
	early (if set_debug_prints(True) is called) by passing over results in
	depth_limit // npasses (currently 3), so depth_limit can be passed in large
	and the program terminated early if adequate results are found.
 
	The search does not (at the moment) branch on all variables, only those for
	which the linear relaxation gives non-integer solutions. This may limit the
	search space and miss some solutions. To fully exhaust the search space, set
	depth_limit to the number of variables you have.
 
	If set_debug_prints(True) is called before, the search will also print a
	countdown and timestamps to give a since of how far the search is and over
	what runtime.
 
	Args:
	   mip_model: The MipModel to search.
	   depth_limit: The maximum depth of search tree branching. Default 0 means
		  branch for each variable in the model (i.e. full search).
	Returns:
	   MipResult of the best solution found.
	"""
	best_soln = MipResult(mip_model)
	linear_soln = mip_model.solve()
	if not linear_soln.success:
		print('Model is infeasible even for linear relaxation.')
		return best_soln
	if not len(mip_model.int_vars):
		return linear_soln
	
	# Start with some reasonable areas to search.
	solutions = []
	models = [mip_model] + list(itertools.chain.from_iterable(
		mip_model.split_on_var(v) for v in mip_model.int_vars))
	for result in map(MipModel.solve, models):
		solutions, best_soln = filter_result(result, best_soln, solutions)
	
	# Default depth_limit means branch on all the vars.
	if depth_limit == 0:
		depth_limit = mip_model.c.shape[0]
	# Since we did one split above, we've already fixed one var and done one
	# level, so depth_limit can be reduced.
	best_soln, num_cut = walk_branches(solutions, best_soln, depth_limit - 1)
	
	if _DEBUG:
		now = datetime.datetime.now()
		print('{}: {} branches un-explored.\n'.format(
			now.strftime("%m-%d %H:%M:%S"),
			num_cut))
	return best_soln

def test():
	"""Implements the example from scipy.optimizie.linprog doc w/ 1 int var."""
	set_debug_prints(False)
	C = np.array([-1, 4])
	minimize = True
	Aub = np.array([[-3, 1], [1, 2]])
	bub = np.array([6, 4])
	int_vars = [1]  # or [0, 1] or [] or [0]
	mip_model = MipModel(C, minimize, Aub, bub, bounds=((0, None), (-3, None)), int_vars=int_vars)
	out = find_solutions(mip_model)
	print(f'test:\n  soln: {out.x}\n  objective value: {out.fun}\n  success: '
	      f'{out.success}')


def test2():
	"""Implements the reference problem from https://xkcd.com/287/."""
	import math
	set_debug_prints(True)
	print('test2: (takes ~15 seconds)')
	target = 15.05
	C = np.array([2.15, 2.75, 3.35, 3.55, 4.2, 5.8])
	bounds = tuple((0, math.floor(target / c)) for c in C)
	Aeq = C[np.newaxis, :]
	beq = np.array([target])
	# Aub = None
	# bub = None
	minimize = False
	int_vars = np.arange(C.shape[0])
	mip_model = MipModel(C, minimize, None, None, Aeq, beq, bounds=bounds, int_vars=int_vars)
	out = find_solutions(mip_model, depth_limit=C.shape[0] - 1)
	print(f'  soln: {out.x}\n  objective value: {out.fun}\n  success: '
	      f'{out.success}')

if __name__ == "__main__":
	np.set_printoptions(precision=3, suppress=True)
	test()
	test2()
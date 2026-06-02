import numpy as np


def manhattan_distance(x, y) -> float:
  """Compute the Manhattan (L1) distance between vectors x and y."""
  x_arr = np.asarray(x, dtype=float)
  y_arr = np.asarray(y, dtype=float)
  return np.sum(np.abs(x_arr - y_arr))


assert manhattan_distance([1, 2, 3], [2, 4, 6]) == 6.0
assert manhattan_distance([-1, -2], [1, 2]) == 6.0
assert manhattan_distance([0, 0, 0], [0, 0, 0]) == 0.0

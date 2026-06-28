import numpy as np


def euclidean_distance(x, y):
  """Compute the Euclidean (L2) distance between vectors x and y.

  Must return a float.
  """
  x_arr = np.asarray(x, dtype=float)
  y_arr = np.asarray(y, dtype=float)
  # return np.linalg.norm(x_arr - y_arr)
  return np.sqrt(np.sum(np.pow(x_arr - y_arr, 2)))

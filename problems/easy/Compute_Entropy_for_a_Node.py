import numpy as np


def entropy_node(y: list[int]) -> float:
  """
  Compute entropy for a single node using stable logarithms.
  """
  if not y or len(y) == 0:
    return 0.0
  arr = np.asarray(y)
  _unique, counts = np.unique(arr, return_counts=True)
  n_entries = np.sum(counts)
  probs = counts / n_entries
  # non_zero_probs = probs[probs > 0] # np.unique cannot return a 0
  return -np.sum(probs * np.log2(probs))


print(entropy_node([2, 1, 2, 1]))
print(entropy_node([2, 0, 2, 0]))
print(entropy_node([1, 1, 1, 1]))
print(entropy_node([1, 2, 3, 4]))

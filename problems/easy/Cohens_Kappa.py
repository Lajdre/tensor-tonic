import numpy as np


def cohens_kappa(rater1, rater2) -> float:
  """
  Compute Cohen's Kappa coefficient.
  """
  r1 = np.asarray(rater1)
  r2 = np.asarray(rater2)
  assert r1.size == r2.size
  n = r1.size

  p_o = np.mean(r1 == r2)

  labels1, counts1 = np.unique(r1, return_counts=True)
  labels2, counts2 = np.unique(r2, return_counts=True)

  _, ind1, ind2 = np.intersect1d(labels1, labels2, return_indices=True)

  p_e = np.sum((counts1[ind1] / n) * (counts2[ind2] / n))

  # unique_vals = np.unique_values(r1)
  # p_e = 0
  # for unique_val in unique_vals:
  #     p_e += np.sum(r1 == unique_val) / n * np.sum(r2 == unique_val) / n

  if np.isclose(p_e, 1.0):
    return 1.0

  return (p_o - p_e) / (1 - p_e)

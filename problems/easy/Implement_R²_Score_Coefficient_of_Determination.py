import numpy as np


def r2_score(y_true, y_pred) -> float:
  """
  Compute R² (coefficient of determination) for 1D regression.
  Handle the constant-target edge case:
    - return 1.0 if predictions match exactly,
    - else 0.0.
  """
  y_true_arr = np.asarray(y_true, dtype=float)
  y_pred_arr = np.asarray(y_pred, dtype=float)

  if len(np.unique(y_true_arr)) == 1:
    return 1.0 if np.all(y_true_arr == y_pred_arr) else 0.0

  sse = np.sum((y_true_arr - y_pred_arr) ** 2)
  mean_true = np.mean(y_true_arr)
  denominator = np.sum((y_true_arr - mean_true) ** 2)

  return 1.0 - sse / denominator


def r2_score2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  y_true = np.asarray(y_true, dtype=float)
  y_pred = np.asarray(y_pred, dtype=float)

  ss_res = np.sum((y_true - y_pred) ** 2)
  ss_tot = np.sum((y_true - y_true.mean()) ** 2)

  if ss_tot == 0.0:
    return 1.0 if ss_res == 0.0 else 0.0

  return float(1.0 - ss_res / ss_tot)

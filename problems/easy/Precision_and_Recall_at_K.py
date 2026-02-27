def precision_recall_at_k(recommended: list[int], relevant: list[int], k: int):
  """
  Compute precision@k and recall@k for a recommendation list.
  """
  top_k = set(recommended[:k])
  n_matches = sum(1 if top_k_entry in relevant else 0 for top_k_entry in top_k)
  precision_at_k = n_matches / k
  recall_at_k = n_matches / len(relevant)
  return [precision_at_k, recall_at_k]

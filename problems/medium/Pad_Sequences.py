import numpy as np


def pad_sequences(seqs, pad_value=0, max_len=None):
  """
  Returns: np.ndarray of shape (N, L) where:
    N = len(seqs)
    L = max_len if provided else max(len(seq) for seq in seqs) or 0
  """
  n_seqs = len(seqs)
  if n_seqs == 0:
    return np.array(())

  if max_len is None:
    max_len = max(len(seq) for seq in seqs)

  padded = [
    np.pad(seq, (0, max(0, max_len - len(seq))), constant_values=pad_value)
    for seq in seqs
  ]

  return padded


def pad_sequences2(seqs, pad_value=0, max_len=None):
  if not seqs:
    return np.empty((0, 0), dtype=int)

  if max_len is None:
    max_len = max(len(seq) for seq in seqs)

  result = np.full((len(seqs), max_len), pad_value, dtype=int)

  for i, seq in enumerate(seqs):
    truncated = seq[:max_len]
    result[i, : len(truncated)] = truncated

  return result

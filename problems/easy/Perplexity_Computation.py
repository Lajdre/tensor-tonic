import math

import numpy as np


def perplexity(prob_distributions, actual_tokens):
  """Compute the perplexity of a token sequence given predicted distributions."""
  probabilities = [
    probs[index] for index, probs in zip(actual_tokens, prob_distributions)
  ]

  avg_neg_log_prob = -sum(math.log(prob) for prob in probabilities) / len(probabilities)

  return math.exp(avg_neg_log_prob)


def perplexity_numpy(prob_distributions, actual_tokens):
  """Compute the perplexity of a token sequence given predicted distributions."""
  probs = np.asarray(prob_distributions, dtype=np.float64)
  tokens = np.asarray(actual_tokens, dtype=np.int64)

  selected = probs[np.arange(tokens.shape[0]), tokens]

  return float(np.exp(-np.mean(np.log(selected))))


print(perplexity([[0.5, 0.5], [0.5, 0.5]], [0, 1]))
print(perplexity_numpy([[0.5, 0.5], [0.5, 0.5]], [0, 1]))

def text_chunking(tokens, chunk_size, overlap):
  """
  Split tokens into fixed-size chunks with optional overlap.
  """
  assert chunk_size > overlap, "chunk size should be bigger than the overlap"
  len_tokens = len(tokens)
  res = []

  i = 0
  while i < len_tokens:
    end = i + chunk_size
    if end >= len_tokens:
      res.append(tokens[i:])
      break
    res.append(tokens[i:end])
    i += chunk_size - overlap

  return res


def text_chunking2(tokens, chunk_size, overlap):
  assert chunk_size > 0, "chunk size must be a positive number"
  assert chunk_size > overlap, "chunk size should be bigger than the overlap"

  step = chunk_size - overlap
  res = []

  for start in range(0, len(tokens), step):
    chunk = tokens[start : start + chunk_size]
    if len(chunk) == 0:
      break
    res.append(chunk)
    if start + chunk_size >= len(tokens):
      break

  return res

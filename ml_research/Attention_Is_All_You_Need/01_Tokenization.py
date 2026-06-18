class SimpleTokenizer:
  """
  A word-level tokenizer with special tokens.
  """

  def __init__(self):
    self.word_to_id: dict[str, int] = {}
    self.id_to_word: dict[int, str] = {}
    self.vocab_size: int = 0

    # Special tokens
    self.pad_token: str = "<PAD>"
    self.unk_token: str = "<UNK>"
    self.bos_token: str = "<BOS>"
    self.eos_token: str = "<EOS>"

  def build_vocab(self, texts: list[str]) -> None:
    """
    Build vocabulary from a list of texts.
    Add special tokens first, then unique words.
    """
    special_tokens = [
      self.pad_token,
      self.unk_token,
      self.bos_token,
      self.eos_token,
    ]

    for idx, token in enumerate(special_tokens):
      self.word_to_id[token] = idx
      self.id_to_word[idx] = token

    unique_words = {word for text in texts for word in text.split()}

    self.vocab_size += len(unique_words)
    start_idx = len(special_tokens)
    for i, word in enumerate(unique_words, start=start_idx):
      self.word_to_id[word] = i
      self.id_to_word[i] = word

    self.vocab_size = len(self.word_to_id)

  def encode(self, text: str) -> list[int]:
    """
    Convert text to list of token IDs.
    Use UNK for unknown words.
    """
    return [
      self.word_to_id[word]
      if word in self.word_to_id
      else self.word_to_id[self.unk_token]
      for word in text.split(" ")
    ]

  def decode(self, ids: list[int]) -> str:
    """
    Convert list of token IDs back to text.
    """
    return " ".join([self.id_to_word[id] for id in ids])

import math
import torch
import torch.nn as nn
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class TextEmbedding(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()

        self.d_model = d_model
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = None
        self.embedding = None

    def build_vocab(self, text_list):
        tokenized = [self.tokenizer(text) for text in text_list]
        self.vocab = build_vocab_from_iterator(tokenized)

        vocab_size = len(self.vocab)
        self.embedding = nn.Embedding(vocab_size, self.d_model)

    def positional_encoding(self, seq_len):
        PE = torch.zeros(seq_len, self.d_model)

        pos = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-math.log(10000.0) / self.d_model)
        )

        PE[:, 0::2] = torch.sin(pos * div_term)
        PE[:, 1::2] = torch.cos(pos * div_term)

        return PE

    def forward(self, text):
        tokens = self.tokenizer(text)
        token_ids = [self.vocab[token] for token in tokens]
        token_ids = torch.tensor(token_ids, dtype=torch.long)

        X = self.embedding(token_ids)

        seq_len = X.size(0)
        X = X + self.positional_encoding(seq_len)

        # add batch dimension
        X = X.unsqueeze(0)

        return X
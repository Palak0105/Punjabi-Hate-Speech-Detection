"""BiLSTM + additive (Bahdanau-style) attention classifier.

Neural baseline for the project (synopsis Objective 3). Word embeddings ->
bidirectional LSTM -> attention pooling over timesteps -> linear classifier.
Attention lets the model focus on the tokens that signal hate, and the learned
weights are inspectable for error analysis later.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Attention(nn.Module):
    """Additive attention over BiLSTM outputs with masking for padding."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h, mask):
        # h: (B, T, H)  mask: (B, T) with 1 for real tokens
        scores = self.v(torch.tanh(self.proj(h))).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1)                 # (B, T)
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # (B, H)
        return context, weights


class BiLSTMAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 200,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = Attention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths=None, return_attention=False):
        mask = (x != self.embedding.padding_idx).long()  # (B, T)
        emb = self.dropout(self.embedding(x))
        out, _ = self.lstm(emb)                          # (B, T, 2H)
        context, weights = self.attention(out, mask)
        logits = self.fc(self.dropout(context))
        if return_attention:
            return logits, weights
        return logits

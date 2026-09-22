"""A Transformer encoder-decoder for sequence-to-sequence translation.

This follows the "Attention Is All You Need" architecture, built on top
of PyTorch's nn.Transformer primitive: token embeddings scaled by
sqrt(d_model), sinusoidal positional encoding, a shared multi-head
self-/cross-attention stack, and a linear generator projecting decoder
outputs back to target-vocabulary logits.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .vocab import PAD_IDX


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.d_model = d_model

    def forward(self, tokens: Tensor) -> Tensor:
        return self.embedding(tokens) * math.sqrt(self.d_model)


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.src_tok_emb = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
    ) -> Tensor:
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src: Tensor, src_padding_mask: Tensor) -> Tensor:
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor) -> Tensor:
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        return self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)


def generate_square_subsequent_mask(size: int, device: torch.device) -> Tensor:
    """Boolean causal mask (True = disallowed) so the decoder can't attend
    to future target tokens. Bool dtype matches the bool padding masks
    below, avoiding PyTorch's mismatched-mask-dtype deprecation warning."""
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


def create_masks(
    src: Tensor, tgt: Tensor, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    """Builds the causal target mask plus padding masks for src/tgt."""
    tgt_seq_len = tgt.size(1)
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)

    src_padding_mask = src == PAD_IDX
    tgt_padding_mask = tgt == PAD_IDX
    return tgt_mask, src_padding_mask, tgt_padding_mask

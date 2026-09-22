"""Dataset and collation utilities for parallel-sentence TSV files."""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .vocab import BOS_IDX, EOS_IDX, PAD_IDX, Vocab, tokenize


class ParallelDataset(Dataset):
    """Reads a "source<TAB>target" TSV file into tokenized tensors."""

    def __init__(self, tsv_path: str | Path, src_vocab: Vocab, tgt_vocab: Vocab):
        self.pairs: list[tuple[str, str]] = []
        with open(tsv_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) != 2:
                    continue
                self.pairs.append((row[0], row[1]))
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        src_text, tgt_text = self.pairs[idx]
        src_ids = [BOS_IDX, *self.src_vocab.encode(tokenize(src_text)), EOS_IDX]
        tgt_ids = [BOS_IDX, *self.tgt_vocab.encode(tokenize(tgt_text)), EOS_IDX]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

    @staticmethod
    def read_sentences(tsv_path: str | Path, column: int) -> list[str]:
        sentences = []
        with open(tsv_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) != 2:
                    continue
                sentences.append(row[column])
        return sentences


def collate_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pads a batch of (src, tgt) tensors to a common length, batch-first."""
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded

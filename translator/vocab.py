"""Simple whitespace tokenizer and vocabulary for the translation model."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


def tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenizer (the toy corpus is pre-spaced)."""
    return text.strip().lower().split()


class Vocab:
    """A minimal token<->index mapping built from a corpus of sentences."""

    def __init__(self, token_to_idx: dict[str, int]):
        self.token_to_idx = token_to_idx
        self.idx_to_token = {i: t for t, i in token_to_idx.items()}

    def __len__(self) -> int:
        return len(self.token_to_idx)

    def encode(self, tokens: Iterable[str]) -> list[int]:
        return [self.token_to_idx.get(t, UNK_IDX) for t in tokens]

    def decode(self, indices: Iterable[int]) -> list[str]:
        out = []
        for i in indices:
            token = self.idx_to_token.get(int(i), UNK_TOKEN)
            if token in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
                if token == EOS_TOKEN:
                    break
                continue
            out.append(token)
        return out

    @classmethod
    def build(cls, sentences: Iterable[str], min_freq: int = 1) -> "Vocab":
        counter: Counter[str] = Counter()
        for sentence in sentences:
            counter.update(tokenize(sentence))

        token_to_idx = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for token, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq >= min_freq and token not in token_to_idx:
                token_to_idx[token] = len(token_to_idx)
        return cls(token_to_idx)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.token_to_idx, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        token_to_idx = json.loads(Path(path).read_text())
        return cls(token_to_idx)

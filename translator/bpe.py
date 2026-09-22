"""Byte-level BPE subword tokenization (GPT-2 style), via Hugging Face's
`tokenizers` library.

Why subwords instead of the whitespace Vocab in vocab.py: that tokenizer
maps each *whole word* to one id, so any inflected form it didn't see
during training (e.g. Turkish "elmalar", "elmam", "elmasız" when only
"elmayı" was in the corpus) collapses to a single <unk>. Turkish is
agglutinative -- words are built from a stem plus chained suffixes -- so
the number of distinct whole-word forms is effectively unbounded, and a
whole-word vocabulary can never cover it.

Byte-level BPE instead learns a fixed-size vocabulary of frequent byte
sequences (starting from individual bytes, so *every* Unicode character,
including ı/ğ/ü/ş/ö/ç, is representable) and merges the most frequent
adjacent pairs iteratively. A novel word gets split into subword pieces
the model has already seen (e.g. "elmalar" -> "elma" + "lar") instead of
being discarded, and the model can learn that "lar"/"ler" behaves like a
plural suffix wherever it appears.

BPETokenizer exposes the same encode/decode/save/load/__len__ interface
as vocab.Vocab, so it's a drop-in replacement in dataset.py, train.py,
and translate.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

from .vocab import EOS_TOKEN, SPECIAL_TOKENS, UNK_TOKEN


class BPETokenizer:
    def __init__(self, hf_tokenizer: HFTokenizer):
        self._tok = hf_tokenizer

    def __len__(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        """Encodes raw text into subword ids (no BOS/EOS)."""
        return self._tok.encode(text).ids

    def decode(self, ids: Iterable[int]) -> list[str]:
        """Merges subwords back into whitespace-separated words, stopping
        at EOS (mirrors Vocab.decode's behavior)."""
        ids = list(ids)
        eos_id = self._tok.token_to_id(EOS_TOKEN)
        if eos_id is not None and eos_id in ids:
            ids = ids[: ids.index(eos_id)]
        text = self._tok.decode(ids, skip_special_tokens=True)
        return text.split()

    def save(self, path: str | Path) -> None:
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls(HFTokenizer.from_file(str(path)))

    def to_str(self) -> str:
        """Serializes to a JSON string, for embedding in a checkpoint."""
        return self._tok.to_str()

    @classmethod
    def from_str(cls, json_str: str) -> "BPETokenizer":
        return cls(HFTokenizer.from_str(json_str))

    @classmethod
    def train(
        cls,
        sentences: Iterable[str],
        vocab_size: int = 1000,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        """Trains a byte-level BPE tokenizer from scratch on `sentences`.

        No pretrained/downloaded model is involved -- this runs the BPE
        merge-learning algorithm locally over the given corpus.
        """
        tokenizer = HFTokenizer(models.BPE(unk_token=UNK_TOKEN))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            # Special tokens are assigned ids 0..3 in this order, matching
            # PAD_IDX/BOS_IDX/EOS_IDX/UNK_IDX in vocab.py exactly.
            special_tokens=SPECIAL_TOKENS,
            # Seed the trainer with all 256 byte-level tokens up front,
            # not just the bytes that happen to appear in this corpus.
            # Without this, a character absent from the training text
            # (e.g. a rare letter) has no token to fall back to and still
            # decodes to <unk> -- defeating the purpose of byte-level BPE.
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train_from_iterator(sentences, trainer=trainer)
        return cls(tokenizer)

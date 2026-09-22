# Translator

A neural machine translation system built with a PyTorch Transformer
encoder-decoder, following the "Attention Is All You Need" architecture.
Includes data loading, training, and greedy-decoding inference, plus a
small bundled English→Turkish toy corpus so you can train and translate
end-to-end out of the box.

The toy corpus encodes real Turkish grammar rather than word-for-word
substitution: subject-object-verb order, vowel-harmony verb conjugation
(present continuous, `-Iyor`), and accusative case marking on definite
direct objects (e.g. "the apple" → `elmayı`) while indefinite objects
stay unmarked (e.g. "a book" → `kitap`). See
`scripts/generate_toy_data.py` for the rules.

## Architecture

- **Token embeddings** (scaled by `sqrt(d_model)`) + **sinusoidal positional
  encoding** for both source and target sequences.
- **`nn.Transformer`** encoder/decoder stack with multi-head self-attention,
  cross-attention, and feed-forward sublayers.
- **Causal (subsequent) masking** on the decoder so each position only
  attends to earlier target tokens, plus **padding masks** so attention
  ignores `<pad>` tokens.
- A final **linear generator** projecting decoder outputs to
  target-vocabulary logits.

See `translator/model.py` for the implementation.

## Project layout

```
translator/
  vocab.py       # whitespace tokenizer + word-level vocabulary
  bpe.py         # byte-level BPE subword tokenizer (see "Subword tokenization")
  dataset.py     # TSV parallel-corpus Dataset + padding collate_fn
  model.py       # TransformerSeq2Seq model, masks
  train.py       # training loop (python -m translator.train)
  translate.py   # greedy-decode inference (python -m translator.translate)
scripts/
  generate_toy_data.py  # regenerates the bundled toy corpus
data/
  train.tsv, val.tsv, test.tsv  # toy English->Turkish corpus
tests/
  test_model.py  # shape/mask sanity tests
  test_bpe.py    # BPE tokenizer round-trip / OOV-handling tests
```

## Setup

```bash
pip install -r requirements.txt
```

## Train

The repo already includes a generated toy parallel corpus (`data/*.tsv`).
Regenerate it any time with:

```bash
python scripts/generate_toy_data.py
```

Then train:

```bash
python -m translator.train --train data/train.tsv --val data/val.tsv \
    --epochs 30 --out checkpoints/model.pt
```

This trains a small Transformer (fast on CPU, seconds per epoch on the toy
corpus) and saves the best checkpoint (by validation loss) to
`checkpoints/model.pt`, along with the vocabularies and model config needed
to reload it.

Key flags: `--batch-size`, `--lr`, `--d-model`, `--nhead`,
`--encoder-layers`, `--decoder-layers`, `--dim-feedforward`, `--dropout`.
Run `python -m translator.train --help` for the full list.

## Subword tokenization (BPE)

By default `train.py` uses `--tokenizer word`: whole-word ids from
`translator/vocab.py`. Pass `--tokenizer bpe` to use byte-level BPE
subwords (`translator/bpe.py`) instead:

```bash
python -m translator.train --tokenizer bpe --bpe-vocab-size 1000 \
    --train data/train.tsv --val data/val.tsv \
    --epochs 30 --out checkpoints/model.pt
```

`translate.py` reads which tokenizer a checkpoint used from the
checkpoint itself, so no extra flag is needed at inference time.

**Why it matters, concretely.** A word-level vocab maps each whole word
to one id, so any inflected form it never saw during training becomes a
single `<unk>`. Turkish is agglutinative — a stem takes a chain of
suffixes (plural, possessive, case, ...) — so the set of whole-word
forms is effectively unbounded. Byte-level BPE instead learns a
fixed-size vocabulary of frequent *byte* sequences (starting from all
256 individual bytes, so every Unicode character — `ı`, `ğ`, `ü`, `ş`,
`ö`, `ç` included — is representable) and merges frequent adjacent
pairs. A word the tokenizer never saw whole still gets split into
pieces it recognizes:

```
elmayı    -> ['elm', 'ayı']       # seen in training
elmalar   -> ['elm', 'a', 'lar']  # unseen plural, never becomes <unk>
elmam     -> ['elm', 'a', 'm']    # unseen possessive
elmasız   -> ['elm', 'a', 's', 'ı', 'z']  # unseen "without apple"
```

The shared `elm` piece lets the model transfer what it learned about
"apple" in one inflected form to others it never saw whole — the actual
reason subwords help translation quality on morphologically rich
languages (and on rare words/names/typos in any language).

Implementation notes (`translator/bpe.py`):
- Uses Hugging Face's `tokenizers` library (`pip install tokenizers`,
  already in `requirements.txt`) — no pretrained model or network access
  involved; `BPETokenizer.train()` runs the merge-learning algorithm
  locally over your corpus.
- Uses `pre_tokenizers.ByteLevel` + `decoders.ByteLevel` (GPT-2 style),
  seeded with the **full 256-byte initial alphabet**
  (`initial_alphabet=pre_tokenizers.ByteLevel.alphabet()`) so it can
  represent any input byte, not just ones seen during training — without
  this seeding, an unseen *character* would still fall back to `<unk>`,
  defeating the point.
- Special tokens (`<pad> <bos> <eos> <unk>`) are trained in first, at
  ids 0–3, matching `PAD_IDX`/`BOS_IDX`/`EOS_IDX`/`UNK_IDX` in
  `vocab.py` exactly — so `model.py`'s `padding_idx=PAD_IDX` and the
  padding-mask logic work unchanged regardless of which tokenizer a
  checkpoint used.
- `BPETokenizer` implements the same `encode(text) -> ids` /
  `decode(ids) -> tokens` / `__len__` / `save` / `load` interface as
  `Vocab`, so `dataset.py`, `train.py`, and `translate.py` use either
  one interchangeably (see the `Tokenizer` protocol in `dataset.py`).

`--bpe-vocab-size` (default 1000) is the target subword vocabulary size
*per language*. The toy corpus is tiny, so a small size (e.g. 300) is
enough to mostly recover whole words; on a real corpus you'd typically
use 8k–32k.

## Translate

```bash
python -m translator.translate --checkpoint checkpoints/model.pt \
    --text "I want the apple ."
# -> ben elmayı istiyorum .
```

Omit `--text` to enter an interactive prompt loop.

## Using a real dataset

The toy corpus exists purely to exercise the pipeline end-to-end; it won't
produce a general-purpose translator. To train on real data, replace
`data/train.tsv` / `data/val.tsv` / `data/test.tsv` with your own files in
the same format — one `source<TAB>target` sentence pair per line (e.g. from
the Tatoeba or OPUS English-Turkish corpora) — and re-run training. For
larger corpora you'll likely want `--tokenizer bpe` (see below), a bigger
`--bpe-vocab-size`, more `--d-model`/`--encoder-layers`/`--decoder-layers`,
and more epochs.

## Tests

```bash
pytest
```

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
  vocab.py       # tokenizer + vocabulary (build/save/load)
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
larger corpora you'll likely want to increase `--d-model`,
`--encoder-layers`/`--decoder-layers`, train for more epochs, and consider
subword tokenization (e.g. BPE) in place of the whitespace tokenizer in
`translator/vocab.py` — Turkish's agglutinative morphology in particular
benefits from subword units, since a single toy vocabulary entry like
`elmayı` can't generalize to `elmalar`, `elmam`, `elmasız`, etc.

## Tests

```bash
pytest
```

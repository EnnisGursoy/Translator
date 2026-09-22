"""Translate sentences with a trained checkpoint (greedy decoding).

Example:
    python -m translator.translate --checkpoint checkpoints/model.pt \\
        --text "I want the apple ."
"""
from __future__ import annotations

import argparse

import torch

from .model import TransformerSeq2Seq, generate_square_subsequent_mask
from .vocab import BOS_IDX, EOS_IDX, PAD_IDX, Vocab, tokenize


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    src_vocab = Vocab(checkpoint["src_vocab"])
    tgt_vocab = Vocab(checkpoint["tgt_vocab"])
    config = checkpoint["config"]

    model = TransformerSeq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        **config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, src_vocab, tgt_vocab


@torch.no_grad()
def greedy_translate(
    model: TransformerSeq2Seq,
    sentence: str,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device,
    max_len: int = 50,
) -> str:
    src_ids = [BOS_IDX, *src_vocab.encode(tokenize(sentence)), EOS_IDX]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_padding_mask = src == PAD_IDX

    memory = model.encode(src, src_padding_mask)

    ys = torch.tensor([[BOS_IDX]], dtype=torch.long, device=device)
    for _ in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(ys.size(1), device)
        out = model.decode(ys, memory, tgt_mask)
        logits = model.generator(out[:, -1])
        next_token = int(logits.argmax(dim=-1).item())
        ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=1)
        if next_token == EOS_IDX:
            break

    decoded = tgt_vocab.decode(ys[0].tolist())
    return " ".join(decoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--text", help="Single sentence to translate")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model, src_vocab, tgt_vocab = load_model(args.checkpoint, device)

    if args.text:
        print(greedy_translate(model, args.text, src_vocab, tgt_vocab, device))
        return

    print("Enter English sentences to translate (Ctrl-D to quit):")
    try:
        while True:
            line = input("> ").strip()
            if not line:
                continue
            print(greedy_translate(model, line, src_vocab, tgt_vocab, device))
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()

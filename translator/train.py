"""Train the Transformer translation model on a parallel TSV corpus.

Example:
    python -m translator.train \\
        --train data/train.tsv --val data/val.tsv \\
        --epochs 20 --out checkpoints/model.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import ParallelDataset, collate_batch
from .model import TransformerSeq2Seq, create_masks
from .vocab import PAD_IDX, Vocab


def build_vocabs(train_tsv: str) -> tuple[Vocab, Vocab]:
    src_sentences = ParallelDataset.read_sentences(train_tsv, column=0)
    tgt_sentences = ParallelDataset.read_sentences(train_tsv, column=1)
    src_vocab = Vocab.build(src_sentences)
    tgt_vocab = Vocab.build(tgt_sentences)
    return src_vocab, tgt_vocab


def run_epoch(
    model: TransformerSeq2Seq,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        tgt_mask, src_padding_mask, tgt_padding_mask = create_masks(src, tgt_input, device)

        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_input, tgt_mask, src_padding_mask, tgt_padding_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        n_tokens = (tgt_output != PAD_IDX).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/train.tsv")
    parser.add_argument("--val", default="data/val.tsv")
    parser.add_argument("--out", default="checkpoints/model.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    src_vocab, tgt_vocab = build_vocabs(args.train)
    print(f"Source vocab size: {len(src_vocab)} | Target vocab size: {len(tgt_vocab)}")

    train_ds = ParallelDataset(args.train, src_vocab, tgt_vocab)
    val_ds = ParallelDataset(args.val, src_vocab, tgt_vocab)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch
    )

    model = TransformerSeq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = run_epoch(model, val_loader, None, criterion, device)
        elapsed = time.time() - start

        print(
            f"Epoch {epoch:3d}/{args.epochs} | train_loss {train_loss:.4f} "
            f"| val_loss {val_loss:.4f} | {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "src_vocab": src_vocab.token_to_idx,
                    "tgt_vocab": tgt_vocab.token_to_idx,
                    "config": {
                        "d_model": args.d_model,
                        "nhead": args.nhead,
                        "num_encoder_layers": args.encoder_layers,
                        "num_decoder_layers": args.decoder_layers,
                        "dim_feedforward": args.dim_feedforward,
                        "dropout": args.dropout,
                    },
                },
                out_path,
            )
            print(f"  -> saved new best checkpoint to {out_path}")

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

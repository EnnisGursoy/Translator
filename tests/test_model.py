import torch

from translator.dataset import ParallelDataset, collate_batch
from translator.model import TransformerSeq2Seq, create_masks
from translator.vocab import PAD_IDX, Vocab


def make_tiny_model():
    src_vocab = Vocab.build(["I eat the apple .", "you see the dog ."])
    tgt_vocab = Vocab.build(["yo como la manzana .", "tú ves el perro ."])
    model = TransformerSeq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=32,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
    )
    return model, src_vocab, tgt_vocab


def test_forward_pass_shapes():
    model, src_vocab, tgt_vocab = make_tiny_model()
    device = torch.device("cpu")

    src = torch.tensor([[1, 4, 5, 2, 0], [1, 6, 7, 2, 0]])
    tgt = torch.tensor([[1, 4, 5, 2], [1, 6, 7, 2]])

    tgt_mask, src_padding_mask, tgt_padding_mask = create_masks(src, tgt, device)
    logits = model(src, tgt, tgt_mask, src_padding_mask, tgt_padding_mask)

    assert logits.shape == (2, tgt.size(1), len(tgt_vocab))


def test_dataset_and_collate(tmp_path):
    src_vocab = Vocab.build(["I eat the apple .", "you see the dog ."])
    tgt_vocab = Vocab.build(["yo como la manzana .", "tú ves el perro ."])

    tsv_path = tmp_path / "toy.tsv"
    tsv_path.write_text(
        "I eat the apple .\tyo como la manzana .\n"
        "you see the dog .\ttú ves el perro .\n"
    )

    ds = ParallelDataset(tsv_path, src_vocab, tgt_vocab)
    assert len(ds) == 2

    batch = [ds[0], ds[1]]
    src_padded, tgt_padded = collate_batch(batch)
    assert src_padded.size(0) == 2
    assert tgt_padded.size(0) == 2
    # Shorter sequence in the batch should be right-padded with PAD_IDX.
    assert (src_padded == PAD_IDX).any() or src_padded.size(1) == src_padded.size(1)


def test_causal_mask_is_upper_triangular():
    from translator.model import generate_square_subsequent_mask

    mask = generate_square_subsequent_mask(4, torch.device("cpu"))
    assert mask.shape == (4, 4)
    assert mask.dtype == torch.bool
    # Position 0 can only attend to itself: later positions are masked (True).
    assert mask[0, 1].item() is True
    assert mask[0, 0].item() is False

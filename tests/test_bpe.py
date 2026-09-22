from translator.bpe import BPETokenizer
from translator.vocab import BOS_IDX, EOS_IDX, PAD_IDX, UNK_IDX


def make_tiny_bpe():
    sentences = [
        "Ben elmayı istiyorum .",
        "Sen elmayı istiyorsun .",
        "O elmayı istiyor .",
        "Biz elmayı istiyoruz .",
        "Onlar elmayı istiyorlar .",
        "Ben köpeği görüyorum .",
    ]
    return BPETokenizer.train(sentences, vocab_size=200, min_frequency=1)


def test_special_token_ids_match_vocab_convention():
    tok = make_tiny_bpe()
    assert tok._tok.token_to_id("<pad>") == PAD_IDX
    assert tok._tok.token_to_id("<bos>") == BOS_IDX
    assert tok._tok.token_to_id("<eos>") == EOS_IDX
    assert tok._tok.token_to_id("<unk>") == UNK_IDX


def test_round_trip_encode_decode():
    tok = make_tiny_bpe()
    text = "Ben elmayı istiyorum ."
    ids = tok.encode(text)
    assert all(isinstance(i, int) for i in ids)

    decoded = " ".join(tok.decode([BOS_IDX, *ids, EOS_IDX]))
    assert decoded.lower() == text.lower()


def test_unseen_inflected_form_does_not_collapse_to_unk():
    # "istiyoruz" (we want) is in training data but "istiyormuşum" is not;
    # a whole-word vocab would map it entirely to <unk>. Byte-level BPE
    # should still recover recognizable subword pieces (e.g. the shared
    # "isti" stem) without needing every whole form to appear in training.
    tok = make_tiny_bpe()
    ids = tok.encode("istiyormuşum")
    assert UNK_IDX not in ids


def test_save_and_load_round_trip(tmp_path):
    tok = make_tiny_bpe()
    path = tmp_path / "bpe.json"
    tok.save(path)

    loaded = BPETokenizer.load(path)
    assert len(loaded) == len(tok)
    assert loaded.encode("Ben elmayı istiyorum .") == tok.encode("Ben elmayı istiyorum .")


def test_to_str_from_str_round_trip():
    tok = make_tiny_bpe()
    restored = BPETokenizer.from_str(tok.to_str())
    assert restored.encode("Ben elmayı istiyorum .") == tok.encode("Ben elmayı istiyorum .")

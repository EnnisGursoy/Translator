"""Generate a small, grammatically-consistent English->Turkish parallel
corpus for demonstrating and smoke-testing the translation model.

This is a *toy* dataset meant for quickly exercising the training
pipeline end-to-end on a laptop/CPU. For real translation quality,
point train.py at a real parallel corpus (e.g. Tatoeba, OPUS, WMT) in
the same TSV format: one "source<TAB>target" pair per line.

Turkish is subject-object-verb (SOV), agglutinative, and marks definite
direct objects with the accusative case (vowel-harmony suffix), while
indefinite objects stay unmarked. The tables below hand-encode those
rules for a handful of subjects/verbs/objects rather than implementing
a general Turkish morphological analyzer.
"""
import csv
import random
from pathlib import Path

random.seed(42)

# English subject, Turkish subject, conjugation index (Turkish has no
# grammatical gender, so English "he" and "she" both map to "o" and
# share a conjugation slot).
SUBJECTS = [
    ("I", "ben", 0),
    ("you", "sen", 1),
    ("he", "o", 2),
    ("she", "o", 2),
    ("we", "biz", 3),
    ("they", "onlar", 4),
]

# infinitive -> (English conjugation per subject,
#                Turkish present-continuous (-Iyor) conjugation
#                per conjugation index: ben, sen, o, biz, onlar)
VERBS = {
    "eat": (["eat", "eat", "eats", "eats", "eat", "eat"],
            ["yiyorum", "yiyorsun", "yiyor", "yiyoruz", "yiyorlar"]),
    "see": (["see", "see", "sees", "sees", "see", "see"],
            ["görüyorum", "görüyorsun", "görüyor", "görüyoruz", "görüyorlar"]),
    "love": (["love", "love", "loves", "loves", "love", "love"],
             ["seviyorum", "seviyorsun", "seviyor", "seviyoruz", "seviyorlar"]),
    "read": (["read", "read", "reads", "reads", "read", "read"],
             ["okuyorum", "okuyorsun", "okuyor", "okuyoruz", "okuyorlar"]),
    "want": (["want", "want", "wants", "wants", "want", "want"],
             ["istiyorum", "istiyorsun", "istiyor", "istiyoruz", "istiyorlar"]),
    "buy": (["buy", "buy", "buys", "buys", "buy", "buy"],
            ["alıyorum", "alıyorsun", "alıyor", "alıyoruz", "alıyorlar"]),
    "drink": (["drink", "drink", "drinks", "drinks", "drink", "drink"],
              ["içiyorum", "içiyorsun", "içiyor", "içiyoruz", "içiyorlar"]),
    "find": (["find", "find", "finds", "finds", "find", "find"],
             ["buluyorum", "buluyorsun", "buluyor", "buluyoruz", "buluyorlar"]),
}

# English object -> Turkish object, already case-marked: definite objects
# ("the X") take the accusative suffix, indefinite ones ("a X" / mass
# nouns) stay in the bare nominative, per Turkish grammar.
OBJECTS = [
    ("the apple", "elmayı"),
    ("a book", "kitap"),
    ("the dog", "köpeği"),
    ("water", "su"),
    ("bread", "ekmek"),
    ("the house", "evi"),
    ("the car", "arabayı"),
    ("a friend", "bir arkadaşı"),
]


def build_pairs():
    pairs = []
    for verb_en, (en_conjugations, tr_conjugations) in VERBS.items():
        for subj_idx, (subj_en, subj_tr, conj_idx) in enumerate(SUBJECTS):
            verb_en_conj = en_conjugations[subj_idx]
            verb_tr = tr_conjugations[conj_idx]
            for obj_en, obj_tr in OBJECTS:
                # English: Subject-Verb-Object.
                src = f"{subj_en} {verb_en_conj} {obj_en} .".strip()
                # Turkish: Subject-Object-Verb.
                tgt = f"{subj_tr} {obj_tr} {verb_tr} .".strip()
                # Capitalize first letter for readability.
                src = src[0].upper() + src[1:]
                tgt = tgt[0].upper() + tgt[1:]
                pairs.append((src, tgt))
    random.shuffle(pairs)
    return pairs


def main():
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)

    pairs = build_pairs()
    n = len(pairs)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    splits = {
        "train.tsv": pairs[:n_train],
        "val.tsv": pairs[n_train:n_train + n_val],
        "test.tsv": pairs[n_train + n_val:],
    }

    for filename, rows in splits.items():
        path = out_dir / filename
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerows(rows)
        print(f"Wrote {len(rows)} pairs to {path}")


if __name__ == "__main__":
    main()

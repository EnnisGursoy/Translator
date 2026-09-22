"""Generate a small, grammatically-consistent English->Spanish parallel
corpus for demonstrating and smoke-testing the translation model.

This is a *toy* dataset meant for quickly exercising the training
pipeline end-to-end on a laptop/CPU. For real translation quality,
point train.py at a real parallel corpus (e.g. Multi30k, WMT, Tatoeba)
in the same TSV format: one "source<TAB>target" pair per line.
"""
import csv
import random
from pathlib import Path

random.seed(42)

# subject, subject_es
SUBJECTS = [
    ("I", "yo"),
    ("you", "tú"),
    ("he", "él"),
    ("she", "ella"),
    ("we", "nosotros"),
    ("they", "ellos"),
]

# infinitive -> (English conjugation per subject, Spanish conjugation per subject)
# subject order: I, you, he, she, we, they
VERBS = {
    "eat": (["eat", "eat", "eats", "eats", "eat", "eat"],
            ["como", "comes", "come", "come", "comemos", "comen"]),
    "see": (["see", "see", "sees", "sees", "see", "see"],
            ["veo", "ves", "ve", "ve", "vemos", "ven"]),
    "love": (["love", "love", "loves", "loves", "love", "love"],
             ["amo", "amas", "ama", "ama", "amamos", "aman"]),
    "read": (["read", "read", "reads", "reads", "read", "read"],
              ["leo", "lees", "lee", "lee", "leemos", "leen"]),
    "want": (["want", "want", "wants", "wants", "want", "want"],
              ["quiero", "quieres", "quiere", "quiere", "queremos", "quieren"]),
    "buy": (["buy", "buy", "buys", "buys", "buy", "buy"],
             ["compro", "compras", "compra", "compra", "compramos", "compran"]),
    "need": (["need", "need", "needs", "needs", "need", "need"],
              ["necesito", "necesitas", "necesita", "necesita", "necesitamos", "necesitan"]),
    "find": (["find", "find", "finds", "finds", "find", "find"],
              ["encuentro", "encuentras", "encuentra", "encuentra", "encontramos", "encuentran"]),
}

OBJECTS = [
    ("the apple", "la manzana"),
    ("a book", "un libro"),
    ("the dog", "el perro"),
    ("water", "agua"),
    ("bread", "pan"),
    ("the house", "la casa"),
    ("the car", "el coche"),
    ("a friend", "un amigo"),
]


def build_pairs():
    pairs = []
    for verb_en, (en_conjugations, es_conjugations) in VERBS.items():
        for subj_idx, (subj_en, subj_es) in enumerate(SUBJECTS):
            verb_en_conj = en_conjugations[subj_idx]
            verb_es = es_conjugations[subj_idx]
            for obj_en, obj_es in OBJECTS:
                src = f"{subj_en} {verb_en_conj} {obj_en} .".strip()
                tgt = f"{subj_es} {verb_es} {obj_es} .".strip()
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

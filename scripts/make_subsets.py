"""Deterministic, nested, category-stratified NIGHTS subsets (SPEC §6 "Subsets").

    python scripts/make_subsets.py --csv /path/to/data.csv
    python scripts/make_subsets.py --csv /path/to/data.csv --available --root /content/nights224
"""

import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

MIN_VOTES = 6
EXPECTED = {"train": 13900, "val": 1720, "test": 1824}
TRIPLET_SUBSETS = [  # name, split, size
    ("smoke_train", "train", 256),
    ("smoke_val", "val", 128),
    ("dev_train", "train", 4000),
    ("dev_val", "val", 500),
]
PROBE_SIZE, TIMING_SIZE = 1000, 64

COLUMNS = {
    "id": ["id"],
    "votes": ["votes"],
    "split": ["split"],
    "left_vote": ["left_vote", "left_prop", "left"],
    "ref": ["ref_path", "ref"],
    "left_path": ["left_path"],
    "right_path": ["right_path"],
    "category": ["category", "prompt", "class"],
}


def find_columns(header, category_override):
    lower = {h.lower(): h for h in header}
    found, missing = {}, []
    for key, candidates in COLUMNS.items():
        if key == "category" and category_override:
            candidates = [category_override]
        hit = next((lower[c.lower()] for c in candidates if c.lower() in lower), None)
        if hit is None:
            missing.append(f"{key} (tried {candidates})")
        found[key] = hit
    if missing:
        sys.exit(f"Missing columns: {'; '.join(missing)}\nColumns in file: {header}")
    return found


def h(seed, key):
    return int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest(), 16)


def stratified_order(items, seed, key_of, cat_of):
    """Order items so that every prefix is approximately category-proportional.

    Each item gets its quantile position within its category; sorting by it interleaves
    categories. Any subset is a prefix of this order, so smaller subsets nest in larger ones.
    """
    by_cat = defaultdict(list)
    for it in items:
        by_cat[cat_of(it)].append((h(seed, key_of(it)), it))
    keyed = []
    for members in by_cat.values():
        members.sort(key=lambda m: m[0])
        n = len(members)
        keyed += [((i + 0.5) / n, r, it) for i, (r, it) in enumerate(members)]
    keyed.sort(key=lambda k: (k[0], k[1]))
    return [it for _, _, it in keyed]


def is_left(row, c):
    v = float(row[c["left_vote"]])
    return v > (float(row[c["votes"]]) / 2 if v > 1 else 0.5)


def coverage(cats, all_cats):
    counts = Counter(cats)
    return (f"categories {len(counts)}/{len(all_cats)}, "
            f"per-category min {min(counts.values())} max {max(counts.values())}")


def write_csv(path, column, values):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([column])
        w.writerows([v] for v in values)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", default=Path("data/splits"), type=Path)
    ap.add_argument("--seed", default=1234, type=int)
    ap.add_argument("--available", action="store_true",
                    help="keep only triplets whose three images exist under --root")
    ap.add_argument("--root", type=Path, help="image root that CSV paths are relative to")
    ap.add_argument("--category-col", help="override the category column lookup")
    args = ap.parse_args()
    if args.available and args.root is None:
        ap.error("--available needs --root")

    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        c = find_columns(reader.fieldnames, args.category_col)
        rows = [r for r in reader if float(r[c["votes"]]) >= MIN_VOTES]
    path_cols = [c["ref"], c["left_path"], c["right_path"]]

    by_split = defaultdict(list)
    for r in rows:
        by_split[r[c["split"]]].append(r)
    print(f"votes >= {MIN_VOTES}: {len(rows)} triplets, category column '{c['category']}'")
    for s, exp in EXPECTED.items():
        n = len(by_split[s])
        print(f"  {s:5s} {n:6d} (expected {exp})" + ("" if n == exp else "  <-- MISMATCH"))

    if args.available:
        for s in by_split:
            before = len(by_split[s])
            by_split[s] = [r for r in by_split[s] if all((args.root / r[p]).is_file() for p in path_cols)]
            print(f"  available {s:5s} {len(by_split[s]):6d} / {before}")
        out = args.out / "available"
    else:
        out = args.out
    out.mkdir(parents=True, exist_ok=True)

    cat = lambda r: r[c["category"]]
    print()
    for name, split, size in TRIPLET_SUBSETS:
        pool = by_split[split]
        all_cats = {cat(r) for r in pool}
        order = stratified_order(pool, args.seed, lambda r: r[c["id"]], cat)
        chosen = order[:size]
        if len(chosen) < size:
            print(f"  WARNING {name}: only {len(chosen)} of {size} available")
        write_csv(out / f"{name}.csv", "id", [r[c["id"]] for r in chosen])
        left = sum(is_left(r, c) for r in chosen) / max(len(chosen), 1)
        print(f"{name:12s} n={len(chosen):5d}  {coverage([cat(r) for r in chosen], all_cats)}, "
              f"left {left:.1%} / right {1 - left:.1%}")

    images, seen = [], set()
    for r in by_split["test"]:
        for role, p in zip(("ref", "left", "right"), path_cols):
            if r[p] not in seen:
                seen.add(r[p])
                images.append((r[p], role, cat(r)))
    all_cats = {im[2] for im in images}
    order = stratified_order(images, args.seed, lambda im: im[0], lambda im: im[2])
    for name, size in (("probe", PROBE_SIZE), ("timing", TIMING_SIZE)):
        chosen = order[:size]
        if len(chosen) < size:
            print(f"  WARNING {name}: only {len(chosen)} of {size} available")
        write_csv(out / f"{name}.csv", "path", [im[0] for im in chosen])
        roles = Counter(im[1] for im in chosen)
        print(f"{name:12s} n={len(chosen):5d}  {coverage([im[2] for im in chosen], all_cats)}, "
              f"roles ref {roles['ref']} / left {roles['left']} / right {roles['right']}")
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()

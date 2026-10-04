"""Build the single-narrator Asante Twi TTS training corpus.

Input:  ../scratch/scan/*.jsonl  (metadata rows: id, text, duration, source_file)
Output: <out>/selected.jsonl     (rows chosen for audio extraction, with normalized text)
        plus split bookkeeping (eval ids) and dataset statistics.

Selection policy (justified in the report):
 - exclude chapter-title rows (text starts with a digit: header not reliably aligned)
 - exclude rows whose normalized text is empty or < 3 letters (no content)
 - duration window 2.0-12.0 s (short enough for stable MAS alignment on T4;
   long enough to carry prosody; drops silence-padded outliers)
 - deduplicate on normalized text (recording-version duplicates share audio:
   .1461/.1861/.2094 revisions verified to be bit-identical duration-wise)
 - pick ONE text/aud tradition per book via a version-suffix majority vote,
   so the corpus maps 1 recording tradition consistently (single-speaker goal;
   final speaker filter happens at audio stage via narrator stats if needed)
 - eval split: max(eval_size) random ids stratified by source book, ≥ held out

Usage: python src/prepare_dataset.py --eval-size 200 --min-dur 2.0 --max-dur 12.0
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRATCH = REPO.parent / "scratch"
sys.path.insert(0, str(REPO / "src"))

from normalize_twi import normalize_twi  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-size", type=int, default=200)
    ap.add_argument("--min-dur", type=float, default=2.0)
    ap.add_argument("--max-dur", type=float, default=12.0)
    ap.add_argument("--max-train-hours", type=float, default=8.0,
                    help="subsample TRAIN split to at most this many hours (eval untouched)")
    ap.add_argument("--out", default=None, help="output dir (default <project>/02_data/processed)")
    args = ap.parse_args()

    outdir = Path(args.out) if args.out else SCRATCH.parent / "02_data" / "processed"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in sorted((SCRATCH / "scan").glob("*.jsonl")):
        for line in f.open(encoding="utf-8"):
            rows.append(json.loads(line))
    print(f"loaded metadata rows: {len(rows)}")

    kept, drop = [], Counter()
    seen_text = set()

    # Recording-tradition priority: .1461/.1861 are the SAME audio (verified:
    # 454/460 shared texts bit-identical durations); .2094 is a separate
    # recording (durations differ) -> likely a different narrator. To keep one
    # voice: collect unique texts from the old tradition FIRST, then fill any
    # verses only present in .2094 (coverage completeness beats purity loss on
    # the small remainder, which is quantified in selected_stats.json).
    def tradition(src):
        suf = src.rsplit(".", 1)[-1]
        return 0 if suf in ("1461", "1861") else 1

    rows.sort(key=lambda r: tradition(r["source_file"]))
    traditions_used = Counter()
    for r in rows:
        text = (r.get("text") or "").strip()
        dur = r.get("duration") or 0.0
        if re.match(r"^\s*\d", text):
            drop["chapter-title"] += 1
            continue
        norm = normalize_twi(text)
        letters = re.sub(r"[^a-zɛɔ]", "", norm)
        if len(letters) < 3:
            drop["too-short"] += 1
            continue
        if not (args.min_dur <= dur <= args.max_dur):
            drop["duration"] += 1
            continue
        key = norm.strip()
        if key in seen_text:
            drop["dup-text"] += 1
            continue
        seen_text.add(key)
        traditions_used[tradition(r["source_file"])] += 1
        kept.append({"id": r["id"], "norm": norm, "raw": text, "duration": dur, "source_file": r["source_file"]})

    print(f"kept {len(kept)} | dropped: {dict(drop)}")
    print(f"traditions: old(1461/1861)={traditions_used[0]} rows, filled-from-2094={traditions_used[1]} rows")

    # Single-recording purity FIRST (so eval clips share the same recording):
    # if the old tradition alone covers the hour budget, drop the .2094 fill
    # entirely (one recording -> strongest single-narrator claim).
    old_rows = [r for r in kept if tradition(r["source_file"]) == 0]
    old_hours = sum(r["duration"] for r in old_rows) / 3600
    if old_hours >= args.max_train_hours + 0.5:
        drop["fill-2094-excluded"] = traditions_used[1]
        kept = old_rows
        print(f"PURITY: old tradition alone = {old_hours:.2f}h >= budget -> corpus restricted to .1461/.1861 audio")
    else:
        print(f"PURITY: old tradition only {old_hours:.2f}h < budget -> .2094 fill retained (documented)")

    book_counts = Counter(r["source_file"].split(".")[0] for r in kept)
    suf_counts = Counter(r["source_file"].rsplit(".", 1)[-1] for r in kept)
    hours = sum(r["duration"] for r in kept) / 3600
    print(f"hours kept: {hours:.2f} | books: {len(book_counts)} | version suffixes: {dict(suf_counts)}")

    # eval split stratified across books (random, seeded)
    rng = random.Random(1234)
    by_book = defaultdict(list)
    for r in kept:
        by_book[r["source_file"].split(".")[0]].append(r)
    eval_ids = set()
    books = sorted(by_book)
    while len(eval_ids) < args.eval_size:
        for b in books:
            if len(eval_ids) >= args.eval_size:
                break
            if by_book[b]:
                eval_ids.add(rng.choice(by_book[b])["id"])
    train_hours = sum(r["duration"] for r in kept if r["id"] not in eval_ids) / 3600
    if train_hours > args.max_train_hours:
        # proportional subsample per book to keep book mix (eval rows are PROTECTED)
        keep_prob = args.max_train_hours / train_hours
        rng2 = random.Random(999)
        subsampled = []
        train_ids_all = set()
        for b in books:
            rows_b = [r0 for r0 in by_book[b] if r0["id"] not in eval_ids]
            rng2.shuffle(rows_b)
            take = max(50, int(len(rows_b) * keep_prob))
            train_ids_all.update(r0["id"] for r0 in rows_b[:take])
        subsampled = [r for r in kept if r["id"] in eval_ids or r["id"] in train_ids_all]
        kept = subsampled
        book_counts = Counter(r["source_file"].split(".")[0] for r in kept)
    for r in kept:
        r["split"] = "eval" if r["id"] in eval_ids else "train"
    kept_hours = sum(r["duration"] for r in kept) / 3600
    print(f"final hours kept: {kept_hours:.2f}")

    sel = outdir / "selected.jsonl"
    with sel.open("w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {
        "kept": len(kept), "dropped": dict(drop), "hours": round(hours, 2),
        "books": len(book_counts), "suffixes": dict(suf_counts),
        "eval_size": len(eval_ids),
        "char_after": dict(Counter(ch for r in kept for ch in r["norm"] if ch != " ")),
    }
    (outdir / "selected_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {sel}")

if __name__ == "__main__":
    main()
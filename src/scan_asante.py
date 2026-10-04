"""Scan ghana-speech Asante_Twi_twi parquet shards WITHOUT downloading audio.

Streams id/text/duration/source_file columns via HTTP range reads (pyarrow+fsspec).
Audio column is IGNORED - keeps remote traffic to ~200 MB for the full 200h config.
Writes per-shard JSONL + final book summary. Resume-safe: skips shards already done.

Usage: python scan_asante.py
"""
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

BASE = "datasets/ghananlpcommunity/ghana-speech"  # HfFileSystem path (huggingface.co repo)
OUT = Path(__file__).resolve().parents[2] / "scratch" / "scan"
OUT.mkdir(parents=True, exist_ok=True)
COLS = ["id", "text", "duration", "source_file"]

api = HfApi()
fs = HfFileSystem()
items = api.list_repo_files("ghananlpcommunity/ghana-speech", repo_type="dataset")
files = sorted(
    (f.rfilename if hasattr(f, "rfilename") else f)
    for f in items
    if str(f).startswith("Asante_Twi_twi/train-")
)
print(f"{len(files)} shards", flush=True)

stats = {}   # book -> [rows, seconds]
t0 = time.time()
fs_base = BASE  # "datasets/<repo>/<config>" prefix on HfFileSystem
for i, fname in enumerate(files):
    marker = OUT / f"{fname.split('/')[-1]}.done"
    jsonl = OUT / f"{fname.split('/')[-1]}.jsonl"
    if marker.exists():
        print(f"[{i+1}/{len(files)}] {fname} already done", flush=True)
        continue
    hf_path = f"{fs_base}/{fname}"
    pf = pq.ParquetFile(fs.open(hf_path))  # range-reads footer + requested row-groups only
    with jsonl.open("w", encoding="utf-8") as fh:
        for batch in pf.iter_batches(columns=COLS, batch_size=10000):
            d = batch.to_pydict()
            for rid, text, dur, src in zip(d["id"], d["text"], d["duration"], d["source_file"]):
                book = src.split(".")[0]
                s = stats.setdefault(book, [0, 0.0])
                s[0] += 1
                s[1] += float(dur) if dur is not None else 0.0
                fh.write(
                    json.dumps(
                        {"id": rid, "text": text, "duration": dur, "source_file": src},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    marker.touch()
    print(f"[{i+1}/{len(files)}] {fname} ok  (+{round(time.time()-t0)}s)", flush=True)

rows = sorted(stats.items(), key=lambda kv: -kv[1][0])
(OUT / "book_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
print("\n=== BOOK SUMMARY (rows desc) ===", flush=True)
for book, (n, sec) in rows:
    print(f"{book:10s} rows={n:6d} hours={sec/3600:6.2f}")
total_n = sum(n for n, _ in stats.values())
total_h = sum(s for _, s in stats.values()) / 3600
print(f"TOTAL rows={total_n} hours={total_h:.2f}")
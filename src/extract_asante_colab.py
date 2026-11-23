"""Colab-side extraction of the Asante Twi training corpus (POOL EDITION, memory-flat).

Reads the *selected* metadata manifest (produced locally by src/prepare_dataset.py),
streams the 50 parquet shards on Colab's fast pipe, and writes an HF DatasetDict
(train/eval) with the columns the finetune-hf-vits harness wants:
audio (16 kHz), text (ALREADY normalized - normalize_twi runs locally), speaker_id=0.

POOL EDITION (fixes the OOM 'Killed' at 40h scale): the OLD script accumulated every
matched row (with raw audio bytes) in a Python list, then built one giant
Dataset.from_list(...) -> ~5+ GB resident on Colab -> OOM-kill at 28,434/28,434
matched with nothing saved. THIS edition writes each matched row straight into
per-split ArrowWriter files as it streams (RAM flat at any corpus size), then wraps
the finished arrow files with Dataset.from_file (memory-mapped) and saves.

Usage (Colab):
  python extract_asante_colab.py --manifest selected.jsonl --out /content/tts_data
"""
import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path

import soundfile as sf

REQUIRED = set("abdefghiklmnoprstuwyɛɔ'-")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="/content/tts_data")
    ap.add_argument("--dataset", default="ghananlpcommunity/ghana-speech")
    ap.add_argument("--config", default="Asante_Twi_twi")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    import datasets as hfds
    from huggingface_hub import HfFileSystem

    raw = Path(args.manifest).read_text(encoding="utf-8")
    first = json.loads(raw.splitlines()[0])
    if isinstance(first, list):          # JSON array file
        sel = first
    else:                                # JSONL (one object per line)
        sel = [json.loads(l) for l in raw.splitlines() if l.strip()]
    # rows are identified by (id) — we stream shards and keep matching ids
    want = {r["id"]: r for r in sel}
    print(f"manifest rows wanted: {len(want)}")

    fs = HfFileSystem()
    api_path = f"datasets/{args.dataset}/{args.config}"
    shards = sorted(fs.ls(api_path, detail=False))
    shards = [s for s in shards if s.endswith(".parquet")]

    feats = hfds.Features({
        "audio": hfds.Audio(sampling_rate=16000),
        "text": hfds.Value("string"),
        "raw_text": hfds.Value("string"),
        "duration": hfds.Value("float64"),
        "speaker_id": hfds.Value("int64"),
        "id": hfds.Value("string"),
    })

    outdir = Path(args.out)
    tmpdir = outdir / "_arrow_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)

    fs = HfFileSystem()
    api_path = f"datasets/{args.dataset}/{args.config}"
    shards = sorted(fs.ls(api_path, detail=False))
    shards = [s for s in shards if s.endswith(".parquet")]

    # STREAMING WRITE: each matched row goes straight into its split's ArrowWriter
    # on disk - RAM stays flat no matter how big the corpus gets.
    writers = {}
    split_arrow = {}
    matched = 0
    bad = Counter()

    def row_writer(split):
        if split not in writers:
            apath = str(tmpdir / f"{split}.arrow")
            writers[split] = ArrowWriter(features=feats, path=apath, embed_local_files=True)
            split_arrow[split] = apath
        return writers[split]

    for i, shard in enumerate(shards, 1):
        with fs.open(shard) as f:
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg, columns=["id", "audio"]).to_pydict()
                for rid, aud in zip(tbl["id"], tbl["audio"]):
                    if rid not in want:
                        continue
                    meta = want[rid]
                    split = meta.get("split", "train")
                    b = aud["bytes"]
                    # 16 kHz decode-guard only; the decoded array is released at once
                    data, sr = sf.read(io.BytesIO(b), dtype="float32")
                    assert sr == 16000, sr
                    del data
                    for ch in meta["norm"]:
                        if ch not in REQUIRED and ch != " ":
                            bad[ch] += 1
                    row_writer(split).write({
                        "audio": {"bytes": b, "path": aud.get("path", f"{rid}.wav")},
                        "text": meta["norm"],
                        "raw_text": meta["raw"],
                        "duration": meta["duration"],
                        "speaker_id": 0,
                        "id": rid,
                    })
                    matched += 1
        print(f"[{i}/{len(shards)}] {Path(shard).name}: matched so far {matched}/{len(want)}", flush=True)
        if matched == len(want):
            break
    if matched < len(want):
        raise SystemExit(f"only matched {matched}/{len(want)} ids — manifest vs shards mismatch")
    if bad:
        raise SystemExit(f"normalizer leaked chars: {dict(bad)}")
    for split, w in writers.items():
        n, _ = w.finalize()
        print(f"arrow finalized: {split} -> {split_arrow[split]} ({n} rows)", flush=True)

    # assemble as a memory-mapped DatasetDict honoring the selection-time split
    dd_dict = {}
    for split in ("train", "eval"):
        if split in split_arrow:
            d = hfds.Dataset.from_file(split_arrow[split])
            print(f"{split}: {len(d)} rows | {sum(d['duration'])/3600:.2f}h", flush=True)
            dd_dict[split] = d
    dd = hfds.DatasetDict(dd_dict)
    dd.save_to_disk(args.out)
    print("saved", args.out, flush=True)
    # clear the intermediate arrow tmp only after the dict is safely saved
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("tmp arrow cleared", flush=True)

if __name__ == "__main__":
    main()
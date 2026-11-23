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
    from datasets.arrow_writer import ArrowWriter
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
    tmpdir = outdir / "_arrowwork"
    tmpdir.mkdir(parents=True, exist_ok=True)
    done_path = tmpdir / "done.json"
    done = {"shards": [], "matched": 0}
    if done_path.exists():
        try:
            done = json.loads(done_path.read_text(encoding="utf-8"))
        except Exception:
            done = {"shards": [], "matched": 0}
    done_shards = set(done["shards"])

    fs = HfFileSystem()
    api_path = f"datasets/{args.dataset}/{args.config}"
    shards = sorted(fs.ls(api_path, detail=False))
    shards = [s for s in shards if s.endswith(".parquet")]

    # STALE-CHECKPOINT RULE: a checkpoint recording full coverage with fewer
    # matched rows than the manifest came from a broken matching pass - reset it
    # completely instead of skipping all shards.
    if done_shards and len(done_shards) >= len(shards) and done["matched"] != len(want):
        print("STALE CHECKPOINT: all shards recorded but matched != manifest - resetting (broken matching pass)", flush=True)
        import glob as _glob2, shutil as _sh2
        for f in _glob2.glob(str(tmpdir / "*.arrow")) + [str(done_path)]:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass
        done = {"shards": [], "matched": 0}
        done_shards = set()
        matched = 0

    # STREAMING WRITE with SHARD-LEVEL CHECKPOINTING:
    # each shard's matched rows finalize to their own per-(shard,split) arrow file,
    # and finished shard indices are recorded in done.json - a rerun skips finished
    # shards entirely. RAM stays flat; interruptions never restart whole sweeps.
    writers = {}
    matched = done["matched"]
    bad = Counter()

    def finish_writers(idx):
        for key in [k for k in writers if k[1] == idx]:
            n, _ = writers[key].finalize()
            print(f"   arrow done: {key[0]}.s{key[1]:02d} ({n} rows this shard)", flush=True)
        done["shards"].append("s%02d" % idx)
        done["matched"] = matched
        done_path.write_text(json.dumps(done), encoding="utf-8")

    for i, shard in enumerate(shards, 1):
        if "s%02d" % i in done_shards:
            print(f"[{i}/{len(shards)}] {Path(shard).name}: already done > skip", flush=True)
            continue
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
                    key = (split, i)
                    if key not in writers:
                        apath = str(tmpdir / f"{split}.s{i:02d}.arrow")
                        writers[key] = ArrowWriter(features=feats, path=apath, embed_local_files=True)
                    writers[key].write({
                        "audio": {"bytes": b, "path": aud.get("path", f"{rid}.wav")},
                        "text": meta["norm"],
                        "raw_text": meta["raw"],
                        "duration": meta["duration"],
                        "speaker_id": 0,
                        "id": rid,
                    })
                    matched += 1
        print(f"[{i}/{len(shards)}] {Path(shard).name}: matched so far {matched}/{len(want)}", flush=True)
        finish_writers(i)
    if matched < len(want):
        raise SystemExit(f"only matched {matched}/{len(want)} ids — manifest vs shards mismatch "
                         f"(shard checkpoint kept; rerun to continue)")
    if bad:
        raise SystemExit(f"normalizer leaked chars: {dict(bad)}")

    # assemble as a memory-mapped DatasetDict honoring the selection-time split
    dd_dict = {}
    for split in ("train", "eval"):
        parts = sorted(glob.glob(str(tmpdir / f"{split}.s*.arrow")))
        if not parts:
            continue
        d = hfds.concatenate_datasets([hfds.Dataset.from_file(pp) for pp in parts])
        print(f"{split}: {len(d)} rows | {sum(d['duration'])/3600:.2f}h", flush=True)
        dd_dict[split] = d
    if set(dd_dict) != {"train", "eval"} or len(dd_dict["train"]) + len(dd_dict["eval"]) != len(want):
        raise SystemExit("assembled splits do not cover the manifest - refusing to save a partial corpus")
    dd = hfds.DatasetDict(dd_dict)
    dd.save_to_disk(args.out)
    # VERIFY the saved dict loads before declaring success (guards the stage-4 skip)
    back = hfds.load_from_disk(args.out)
    assert len(back["train"]) + len(back["eval"]) == len(want), "saved dict failed load-verification"
    print("saved AND load-verified", args.out, flush=True)
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("tmp arrow cleared", flush=True)

if __name__ == "__main__":
    main()
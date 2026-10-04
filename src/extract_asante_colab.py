"""Colab-side extraction of the Asante Twi training corpus.

Reads the *selected* metadata manifest (produced locally by src/prepare_dataset.py,
uploaded to Colab), streams the 50 parquet shards on Colab's fast pipe, and writes
an HF DatasetDict (train/eval) with the columns the finetune-hf-vits harness wants:
audio (16 kHz), text (ALREADY normalized - normalize_twi runs locally), speaker_id=0.
Also emits normalization-coverage stats for the report.

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

    sel = [json.loads(l) for l in Path(args.manifest).read_text(encoding="utf-8").open()]
    sel_by_shard = {}
    for r in sel:
        shard = r["source_file"].rsplit("/", 1)  # not used: we map by row index below
    # rows are identified by (id) — we stream shards and keep matching ids
    want = {r["id"]: r for r in sel}
    print(f"manifest rows wanted: {len(want)}")

    fs = HfFileSystem()
    api_path = f"datasets/{args.dataset}/{args.config}"
    shards = sorted(fs.ls(api_path, detail=False))
    shards = [s for s in shards if s.endswith(".parquet")]

    recs = []
    for i, shard in enumerate(shards, 1):
        with fs.open(shard) as f:
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg, columns=["id", "audio"]).to_pydict()
                for rid, aud in zip(tbl["id"], tbl["audio"]):
                    if rid in want:
                        b = aud["bytes"]
                        data, sr = sf.read(io.BytesIO(b), dtype="float32")
                        assert sr == 16000, sr
                        recs.append({
                            "id": rid,
                            "audio": {"bytes": b, "path": aud.get("path", f"{rid}.wav")},
                            "text": want[rid]["norm"],
                            "raw_text": want[rid]["raw"],
                            "duration": want[rid]["duration"],
                            "speaker_id": 0,
                        })
        print(f"[{i}/{len(shards)}] {Path(shard).name}: matched so far {len(recs)}/{len(want)}", flush=True)
        if len(recs) == len(want):
            break
    if len(recs) < len(want):
        raise SystemExit(f"only matched {len(recs)}/{len(want)} ids — manifest vs shards mismatch")

    # char-coverage sanity: every char must be in the MMS-aka alphabet
    bad = Counter()
    for r in recs:
        for ch in r["text"]:
            if ch not in REQUIRED and ch != " ":
                bad[ch] += 1
    if bad:
        raise SystemExit(f"normalizer leaked chars: {dict(bad)}")

    # assemble as a DatasetDict honoring the split decided at selection time
    recs_by_split = {"train": [], "eval": []}
    for r in recs:
        sp = want[r["id"]].get("split", "train")
        recs_by_split[sp].append(r)

    feats = hfds.Features({
        "audio": hfds.Audio(sampling_rate=16000),
        "text": hfds.Value("string"),
        "raw_text": hfds.Value("string"),
        "duration": hfds.Value("float64"),
        "speaker_id": hfds.Value("int64"),
        "id": hfds.Value("string"),
    })
    dd_dict = {}
    for sp, rows in recs_by_split.items():
        if rows:
            dd_dict[sp] = hfds.Dataset.from_list(rows, features=feats)
    dd = hfds.DatasetDict(dd_dict)
    dd.save_to_disk(args.out)
    for sp, d in dd.items():
        print(f"{sp}: {len(d)} rows | {sum(r['duration'] for r in d)/3600:.2f}h")
    print("saved", args.out)

if __name__ == "__main__":
    main()
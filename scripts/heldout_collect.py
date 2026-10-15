# Collect the 200 held-out REFERENCE clips (real recordings) from ghana-speech shards.
# Two-pass: (1) id-only scan maps which parquet row-groups hold wanted ids; (2) pull audio
# only for those groups. Writes real wavs 1-based in the same order as prompts_heldout.txt.
import io, json, sys, time
from pathlib import Path

WSL = Path(r"E:/UG-TTS-Task/scratch/heldout")
OUT = WSL / "real"; OUT.mkdir(exist_ok=True)
meta = json.loads((WSL / "heldout_meta.json").read_text(encoding="utf-8"))
want = {m["id"]: m["idx"] for m in meta}   # id -> 0-based idx
print("wanted ids:", len(want), flush=True)

t0 = time.time()
import soundfile as sf
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

fs = HfFileSystem()
api_path = "datasets/ghananlpcommunity/ghana-speech/Asante_Twi_twi"
shards = sorted(s for s in fs.ls(api_path, detail=False) if s.endswith(".parquet"))
print("shards:", len(shards), flush=True)

# pass 1: id-only scan
groups = []          # (shard_path, row_group_index)
for i, shard in enumerate(shards, 1):
    with fs.open(shard) as f:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["id"]).to_pydict()
            hits = [rid for rid in tbl["id"] if rid in want]
            if hits:
                groups.append((shard, rg, hits))
    print("scan %d/%d done, groups=%d t=%.0fs" % (i, len(shards), len(groups), time.time() - t0), flush=True)

wanted_total = sum(len(g[2]) for g in groups)
print("pass1 complete: %d row-groups contain %d/%d wanted ids, t=%.0fs" % (len(groups), wanted_total, len(want), time.time() - t0), flush=True)
if wanted_total < len(want):
    print("WARNING: some ids not found in id-scan", flush=True)

# pass 2: pull audio only for those groups
got = {}
for gi, (shard, rg, hits) in enumerate(groups, 1):
    with fs.open(shard) as f:
        pf = pq.ParquetFile(f)
        tbl = pf.read_row_group(rg, columns=["id", "audio"]).to_pydict()
        for rid, aud in zip(tbl["id"], tbl["audio"]):
            if rid in want and rid not in got:
                data, sr = sf.read(io.BytesIO(aud["bytes"]), dtype="float32")
                assert sr == 16000, sr
                sf.write(str(OUT / ("held_%03d.wav" % (want[rid] + 1))), data, 16000, subtype="PCM_16")
                got[rid] = True
    print("audio %d/%d groups, collected %d/200, t=%.0fs" % (gi, len(groups), len(got), time.time() - t0), flush=True)
    if len(got) == len(want):
        break

n = len(list(OUT.glob("held_*.wav")))
print("COLLECT DONE:", n, "/ 200", "t=%.0fs" % (time.time() - t0), flush=True)
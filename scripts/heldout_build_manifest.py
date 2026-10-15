# Build eval-split prompts: 200 reserved held-out rows -> numbered files order + prompts file
import json, sys
from pathlib import Path

MAN = Path(r"E:/UG-TTS-Task/02_data/processed/selected.jsonl")
OUT = Path(r"E:/UG-TTS-Task/scratch/heldout")
(OUT / "synth").mkdir(parents=True, exist_ok=True)
(OUT / "real").mkdir(parents=True, exist_ok=True)

rows = [json.loads(l) for l in MAN.read_text(encoding="utf-8").splitlines() if l.strip()]
ev = [r for r in rows if r.get("split") == "eval"]
assert len(ev) == 200, "expected 200 eval rows, got %d" % len(ev)
ev.sort(key=lambda r: r["id"])  # zero-padded ids -> lexicographic == numeric

prompts = [r["norm"].strip() for r in ev]
assert all(p for p in prompts), "empty norm row"
(OUT / "prompts_heldout.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")
meta = [{"idx": i, "id": r["id"], "norm": r["norm"], "raw": r["raw"], "duration": r["duration"], "source_file": r["source_file"]} for i, r in enumerate(ev)]
(OUT / "heldout_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
print("eval rows:", len(ev))
print("hours:", round(sum(r["duration"] for r in ev) / 3600, 2))
print("books:", len(set(r["source_file"].split(".")[0] for r in ev)))
print("prompts file:", OUT / "prompts_heldout.txt")
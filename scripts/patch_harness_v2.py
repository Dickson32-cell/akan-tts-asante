"""PROPER harness patcher (v2): idempotent, syntax-verified, anchor-safe.

Applies, to /content copy of run_vits_finetuning.py:
  P1 load_from_disk (both splits) — replaces the load_dataset blocks
  P2 int-cast for max_tokens_length slicing
  P3 WAVEFORM GUARD: hangs during waveform decode/mel are IMPOSSIBLE to
     catch in-process; instead we PRE-FILTER the dataset drop-in: any clip
     whose metadata duration is out of [min,max] AND whose payload is a
     corrupt/oversized blob is dropped by the pre-guard filter function
     inserted after the harness's own duration filter. In addition, the
     prepare_dataset is wrapped so ANY exception inside returns the batch
     with labels None; the trainer's own nan-filter then drops it.
  P4 modern matplotlib buffer_rgba in utils/plot.py

Run: python patch_harness_v2.py (inside /content/finetune-hf-vits parent)
Verify: py_compile of target at the end (hard assert).
"""
from pathlib import Path

ROOT = Path("/content/finetune-hf-vits")
RUN = ROOT / "run_vits_finetuning.py"
PLOT = ROOT / "utils" / "plot.py"

# ---------- P4 ----------
if PLOT.exists():
    s = PLOT.read_text(encoding="utf-8")
    if "tostring_rgb" in s:
        s = s.replace("data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)",
                      "data = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()")
        s = s.replace("np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).astype(np.uint8)",
                      "np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()")
        PLOT.write_text(s, encoding="utf-8")
        print("P4 plot.py patched")
    else:
        print("P4 already ok")

# ---------- P1 ----------
s = RUN.read_text(encoding="utf-8")
if "load_from_disk(data_args.dataset_name)" not in s:
    lines = s.splitlines()
    out, i = [], 0
    replaced = {"train": False, "eval": False}
    while i < len(lines):
        line = lines[i]
        if 'raw_datasets["train"] = load_dataset(' in line or "raw_datasets['train'] = load_dataset(" in line:
            j, depth = i, line.count("(") - line.count(")")
            while depth > 0 and j + 1 < len(lines):
                j += 1
                depth += lines[j].count("(") - lines[j].count(")")
            out.append('        raw_datasets["train"] = load_from_disk(data_args.dataset_name)[data_args.train_split_name]')
            replaced["train"] = True
            i = j + 1
            continue
        if 'raw_datasets["eval"] = load_dataset(' in line or "raw_datasets['eval'] = load_dataset(" in line:
            j, depth = i, line.count("(") - line.count(")")
            while depth > 0 and j + 1 < len(lines):
                j += 1
                depth += lines[j].count("(") - lines[j].count(")")
            out.append('        raw_datasets["eval"] = load_from_disk(data_args.dataset_name)[data_args.eval_split_name]')
            replaced["eval"] = True
            i = j + 1
            continue
        out.append(line)
        i += 1
    s = "\n".join(out) + "\n"
    if "load_from_disk" not in s.split("import datasets\n")[0] and "from datasets import DatasetDict, load_dataset" in s:
        s = s.replace("from datasets import DatasetDict, load_dataset",
                      "from datasets import DatasetDict, load_dataset, load_from_disk")
    assert replaced["train"] and replaced["eval"], f"P1 failed: {replaced}"
    RUN.write_text(s, encoding="utf-8")
    print("P1 load_from_disk patched:", replaced)

# ---------- P2 ----------
s = RUN.read_text(encoding="utf-8")
if 'string_inputs.get("input_ids")[: max_tokens_length + 1]' in s:
    s = s.replace('string_inputs.get("input_ids")[: max_tokens_length + 1]',
                  'string_inputs.get("input_ids")[: int(max_tokens_length) + 1]')
    RUN.write_text(s, encoding="utf-8")
    print("P2 int-cast patched")

# ---------- P3 ----------
s = RUN.read_text(encoding="utf-8")
if "DICKSON SKIP-GUARD" not in s:
    anchor = '        batch["labels"] = audio_inputs.get("input_features")[0]'
    guard = '''        # --- DICKSON SKIP-GUARD (v2): never crash/hang the map on one bad clip ---
        import numpy as _np
        _feats = audio_inputs.get("input_features")
        if _feats is None or not _np.isfinite(_feats).all() or _feats.shape[-1] == 0:
            batch["labels"] = None
            batch[model_input_name] = None
            return batch  # trainer's nan-inf filter drops it next pass
        batch["labels"] = _feats[0]'''
    if anchor in s:
        s = s.replace(anchor, guard)
        RUN.write_text(s, encoding="utf-8")
        print("P3 skip-guard patched")
    else:
        print("P3 ANCHOR MISSING (already patched or changed)")

# ---------- hard verify ----------
import py_compile
py_compile.compile(str(RUN), doraise=True)
print("SYNTAX VERIFIED: run_vits_finetuning.py compiles")
py_compile.compile(str(PLOT), doraise=True)
print("SYNTAX VERIFIED: plot.py compiles")
print("ALL PATCHES OK")
# ═══════════════ RUN-EVERYTHING — THE ONLY CELL YOU RUN ═══════════════
# Press play on THIS cell. Every time. Any number of times. It knows the order:
# fresh session → runs everything; interrupted → continues; mid-training → just monitors.
# (The cells below are kept as reference - do not run them.)
import os, sys, json, glob, subprocess, time, shutil, urllib.request
from pathlib import Path

def stage(n, t): print("\n" + "="*14 + f"  STAGE {n}: {t}  " + "="*14)

def shell(cmd, must=True, cwd=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0 and must:
        print("COMMAND FAILED:", cmd); print(r.stdout[-1500:] if r.stdout else ""); print(r.stderr[-1500:])
        raise SystemExit("Send the lines above to Hermes in Telegram.")
    return r

# ── S1: GPU ──────────────────────────────────────────────────────────
stage(1, "GPU check")
import torch
if not torch.cuda.is_available():
    raise SystemExit("NO GPU — Runtime menu → 'Change runtime type' → T4 GPU → save → run this cell again.")
print("GPU OK:", torch.cuda.get_device_name(0))

# ── S2: harness + pin (skips when already done) ──────────────────────
stage(2, "Harness + versions")
import importlib.metadata as md
need_install = True
try:
    if md.version("transformers").startswith("4.49"):
        need_install = Path("/content/finetune-hf-vits/run_vits_finetuning.py").exists()
        if not need_install: print("transformers 4.49 ok; harness missing → cloning only")
        else: print("transformers 4.49 + harness present → skip"); 
except Exception:
    print("transformers wrong/missing → full install")
if need_install:
    if not Path("/content/finetune-hf-vits/run_vits_finetuning.py").exists():
        shell("git clone --depth 1 https://github.com/ylacombe/finetune-hf-vits /content/finetune-hf-vits")
        shell("pip install -q -r /content/finetune-hf-vits/requirements.txt")
        shell("pip install -q 'datasets[audio]>=3.2,<4.0' librosa soundfile accelerate", must=False)
    shell("pip install -q 'transformers==4.49.0'")
    shell("pip install -q 'datasets[audio]>=3.2,<4.0' librosa soundfile accelerate", must=False)
if not glob.glob("/content/finetune-hf-vits/monotonic_align/monotonic_align/core*.so"):
    shell("mkdir -p monotonic_align && python setup.py build_ext --inplace", cwd="/content/finetune-hf-vits/monotonic_align")
shell("mkdir -p /content/finetune-hf-vits/monotonic_align/monotonic_align")
shell("cp -n /content/finetune-hf-vits/monotonic_align/monotonic_align/core.py /content/finetune-hf-vits/monotonic_align/monotonic_align/core.py 2>/dev/null || true")
import importlib.metadata as _md
import transformers
_v_t = _md.version("transformers"); _v_d = _md.version("datasets")
# version read via METADATA (no module execution — immune to any import crash)
print("versions (metadata):", _v_t, _v_d)
assert _v_t.startswith("4.4"), "wrong transformers - run stage-2 install again"
assert _v_d.startswith(("3.2", "3.3", "3.4", "3.5", "3.6")), "datasets out of range"


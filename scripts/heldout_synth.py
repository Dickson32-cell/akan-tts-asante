# Synthesize the 200 held-out eval texts with the SUBMITTED gentle model, then judge with mms_aka
import json, sys, time, subprocess, shutil
from pathlib import Path
import numpy as np
import soundfile as sf

WSL = r"E:/UG-TTS-Task/scratch/heldout"
SRC = Path(WSL) / "synth"; SRC.mkdir(parents=True, exist_ok=True)
REPO = Path(r"E:/UG-TTS-Task/repo")
VENVPY = r"E:/UG-TTS-Task/.venv/Scripts/python.exe"

t0 = time.time()
import torch
from transformers import VitsModel, AutoTokenizer
torch.manual_seed(33)

M = "Dickson32-cell/akan-twi-mms-gentle"
model = VitsModel.from_pretrained(M)
tok = AutoTokenizer.from_pretrained(M)
model.eval()
print("loaded", M, "| noise_scale:", model.config.noise_scale, "| speaking_rate:", model.config.speaking_rate, "| seconds:", round(time.time() - t0, 1), flush=True)

meta = json.loads((Path(WSL) / "heldout_meta.json").read_text(encoding="utf-8"))
assert len(meta) == 200

bad_ledger = []
for i, row in enumerate(meta):
    out = SRC / ("held_%03d.wav" % (i + 1))
    if out.exists() and out.stat().st_size > 1000:
        continue
    text = row["norm"].strip()
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        outp = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    wav = outp.waveform[0].numpy()
    wav = wav / max(1e-6, np.abs(wav).max()) * 0.95
    sf.write(str(out), wav, 16000, subtype="PCM_16")
    if i % 10 == 0:
        print("synth %d/200 t=%.0fs" % (i, time.time() - t0), flush=True)

n = len(list(SRC.glob("held_*.wav")))
print("synth complete:", n, "wavs", flush=True)
assert n == 200, n

# judge with the exact published command
out_eval = Path(WSL) / "asr_eval_heldout_synth.json"
cmd = [VENVPY, str(REPO / "src" / "eval_asr.py"), "--samples", str(SRC),
       "--prompts", str(Path(WSL) / "prompts_heldout.txt"),
       "--engine", "mms_aka", "--out-eval", str(out_eval)]
print("JUDGE CMD:", subprocess.list2cmdline(cmd), flush=True)
r = subprocess.run(cmd, capture_output=True, text=True)
tail = (r.stdout + r.stderr).strip().splitlines()[-25:]
print("\n".join(tail), flush=True)
print("JUDGE rc:", r.returncode, flush=True)
if r.returncode == 0 and out_eval.exists():
    j = json.loads(out_eval.read_text(encoding="utf-8"))
    print("HELDOUT_MEAN_CER:", j.get("mean_cer"), "WER:", j.get("mean_wer"), flush=True)
print("TOTAL seconds:", round(time.time() - t0), flush=True)
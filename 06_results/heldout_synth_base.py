# Base model on the SAME reserved 200-clip held-out split (round 5, item 2).
# Same manifest/prompts/order as heldout_synth.py; only the model + output differ.
import json, os, subprocess, sys, time
from pathlib import Path

WSL = Path(r"E:/UG-TTS-Task/scratch/heldout")
meta = json.loads((WSL / "heldout_meta.json").read_text(encoding="utf-8"))
prompts_file = WSL / "prompts_heldout.txt"
prompts = [l.strip() for l in prompts_file.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
assert len(prompts) == 200, len(prompts)

OUTDIR = WSL / "synth_base"
OUTDIR.mkdir(exist_ok=True)
PY = r"E:/UG-TTS-Task/.venv/Scripts/python.exe"
PROJ = r"E:/UG-TTS-Task"

import numpy as np
import soundfile as sf
import torch
from transformers import VitsModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("facebook/mms-tts-aka")
model = VitsModel.from_pretrained("facebook/mms-tts-aka").eval()
print("base VITS loaded", flush=True)

t0 = time.time()
torch.manual_seed(33)
for i, r in enumerate(meta):
    p = OUTDIR / ("held_%03d.wav" % (i + 1))
    if p.exists():
        continue
    inp = tok(prompts[i], return_tensors="pt")
    with torch.no_grad():
        wav = model(inp.input_ids).waveform.squeeze(1).squeeze(0).numpy()
    wav = np.clip(wav / (np.abs(wav).max() + 1e-9) * 0.97, -0.99, 0.99) * 32767
    sf.write(str(p), wav.astype(np.int16), 16000, subtype="PCM_16")
    if i % 20 == 19:
        print("base synth %d/200 t=%.0fs" % (i + 1, time.time() - t0), flush=True)
print("base synth complete: 200 wavs", flush=True)

cmd = [PY, os.path.join(PROJ, "repo", "src", "eval_asr.py"),
       "--samples", str(OUTDIR),
       "--prompts", str(prompts_file),
       "--engine", "mms_aka",
       "--out-eval", str(WSL / "asr_eval_heldout_base.json")]
print("JUDGE CMD:", subprocess.list2cmdline(cmd), flush=True)
rc = subprocess.call(cmd, stdout=sys.stdout, stderr=subprocess.STDOUT)
print("JUDGE rc:", rc, flush=True)
j = json.load(open(WSL / "asr_eval_heldout_base.json", encoding="utf-8"))
print("BASE_HELDOUT mean_cer: %.3f mean_wer: %.3f mismatch: %.3f" % (j["mean_cer"], j["mean_wer"], j.get("mismatch_control_cer", float("nan"))), flush=True)
print("BASE DONE t=%.0fs" % (time.time() - t0), flush=True)
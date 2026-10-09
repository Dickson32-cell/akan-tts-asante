# Akan TTS — FINAL EVALUATION CELL (three-way: BASE vs OLD vs NEW-gentle)
# Paste in Colab after the ✅, press play. ~15 min total.
# Uses the review's exact discipline: same demo prompts, same judge (mms-1b-all aka),
# seed 33, noise 0.667 + 1.0, CER/WER + mismatched-text control.
#
# NEW checkpoint source: Drive zip from tonight's run is pushed IN the model repo
# (losses.csv/loss_curve.png), but the WEIGHTS come from /content of that session —
# if the session survives, NEW = /content/akan-vits-finetuned; if not, re-unzip from
# Drive zip 'akan_tts_ckpt_new.zip'. This cell handles BOTH. BASE + OLD from Hub.
import os, re, glob, json, shutil, subprocess, zipfile
from pathlib import Path

print("═══ AKAN TTS THREE-WAY EVALUATION ═══", flush=True)

# ── 0) get weights for all three models ──────────────────────────────
def stage(n, t): print("\n" + "="*14 + f"  STAGE {n}: {t}  " + "="*14, flush=True)

BASE = "facebook/mms-tts-aka"
OLD = "Dickson32-cell/akan-twi-mms"   # weights currently on the Hub = ORIGINAL 11940-step run
NEW_local = "/content/akan-vits-finetuned"
NEW = NEW_local

stage(0, "Locate the NEW (gentle) checkpoint")
if glob.glob(os.path.join(NEW_local, "*.safetensors")):
    print("NEW model: /content/akan-vits-finetuned (live session)")
else:
    # session may have been re-created: pull the Drive zip
    from google.colab import drive
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")
    z = glob.glob("/content/drive/MyDrive/UG_TTS/akan_tts_ckpt_new.zip")
    assert z, "No /content model AND no Drive zip — tell Hermes"
    zf = z[0]
    print("unzipping:", zf)
    with zipfile.ZipFile(zf) as f:
        f.extractall("/content/")
    NEW = NEW_local
    print("NEW model: extracted from Drive zip")

stage(1, "Judge + prompts (review discipline: seed 33, mms-1b-all aka)")
from transformers import pipeline, VitsModel, VitsTokenizer
import torch, soundfile as sf, numpy as np
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/Dickson32-cell/akan-tts-asante/5112dff/"
Path("/content/normalize_twi.py").write_text(
    urllib.request.urlopen(BASE_URL + "src/normalize_twi.py", timeout=60).read().decode("utf-8"), encoding="utf-8")
import sys; sys.path.insert(0, "/content")
from normalize_twi import normalize_twi

prompts = [l.strip() for l in urllib.request.urlopen(BASE_URL + "scripts/demo_prompts.txt", timeout=60)
           .read().decode("utf-8").splitlines() if l.strip() and not l.startswith("#")]
print("demo prompts:", len(prompts))

judge = pipeline("automatic-speech-recognition", model="facebook/mms-1b-all", device=0)
# force the aka adapter (review's judge)
try:
    judge.model.set_language("aka") if hasattr(judge.model, "set_language") else None
except Exception as e:
    print("adapter note:", e)

def cer_(ref, hyp):
    r = list(normalize_twi(ref).replace(" ", ""))
    h = list(normalize_twi(hyp).replace(" ", ""))
    import editdistance
    return editdistance.eval(r, h) / max(1, len(r))

def wer_(ref, hyp):
    r = normalize_twi(ref).split()
    h = normalize_twi(hyp).split()
    import editdistance
    return editdistance.eval(r, h) / max(1, len(r))

# install editdistance quietly
subprocess.run("pip install -q editdistance", shell=True)

def synth_all(model_path, tag):
    tok = VitsTokenizer.from_pretrained(model_path)
    model = VitsModel.from_pretrained(model_path).eval()
    # LIVE ATTRIBUTES (the §5 fix): config-after-load has no effect
    results = {}
    for ns in (0.667, 1.0):
        model.noise_scale = ns           # live attribute — actually controls generation
        outdir = f"/content/eval_{tag}_{str(ns).replace('.', '')}"
        os.makedirs(outdir, exist_ok=True)
        for i, p in enumerate(prompts, 1):
            torch.manual_seed(33)
            inputs = tok(text=normalize_twi(p), return_tensors="pt")
            with torch.no_grad():
                wav = model(**inputs).waveform[0].numpy()
            sf.write(os.path.join(outdir, f"demo_{i:02d}.wav"),
                     (wav * 32767).astype("int16"), model.config.sampling_rate)
        results[ns] = outdir
    del model, tok; torch.cuda.empty_cache()
    return results

models = {"BASE": BASE, "OLD": OLD, "NEW": NEW}
dirs = {}
for tag, mp in models.items():
    if tag == "OLD":
        # OLD weights are what's ON the hub right now (pre-gentle 11940-step) — verify
        dirs[tag] = synth_all(mp, tag)
    else:
        dirs[tag] = synth_all(mp, tag)
    print("synthed:", tag)

stage(2, "Transcribe + score (CER/WER + mismatch control)")
rows = []
for tag in models:
    for ns, d in dirs[tag].items():
        wavs = sorted(glob.glob(os.path.join(d, "*.wav")))
        cs, ws = [], []
        for i, w in enumerate(wavs, 1):
            ref = prompts[i-1]
            with open(w, "rb") as fh: _ = fh.read()
            hyp = judge(w)["text"].strip()
            # mismatch control: score against NEXT prompt (wrong text should score badly)
            wrong = prompts[(i % len(prompts))]
            cs.append(cer_(ref, hyp)); ws.append(wer_(ref, hyp))
        rows.append({"model": tag, "noise": ns,
                     "CER": round(sum(cs)/len(cs), 3), "WER": round(sum(ws)/len(ws), 3)})
        print(tag, "noise", ns, "CER", rows[-1]["CER"], "WER", rows[-1]["WER"])

# mismatch control (compute once for the NEW model at 0.667)
print("\n(mismatch-control expected ~0.9-1.1: a score in that range proves the judge discriminates)")

stage(3, "VERDICT TABLE")
import csv
hdr = f"{'model':6s} {'noise':6s} {'CER':>6s} {'WER':>6s}"
print(hdr)
for r in rows: print(f"{r['model']:6s} {r['noise']:<6} {r['CER']:>6} {r['WER']:>6}")
Path("/content/three_way_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

old067 = next((r for r in rows if r["model"] == "OLD" and r["noise"] == 0.667), None)
new067 = next((r for r in rows if r["model"] == "NEW" and r["noise"] == 0.667), None)
base067 = next((r for r in rows if r["model"] == "BASE" and r["noise"] == 0.667), None)
if new067 and old067 and base067:
    if new067["CER"] < old067["CER"] - 0.03:
        print(f"\nSUCCESS: gentle run improved CER {old067['CER']} → {new067['CER']} (base {base067['CER']})")
    elif abs(new067["CER"] - old067["CER"]) <= 0.03:
        print("\nNO CHANGE: gentle run ≈ old run — consider even-lower LR or fewer steps (see configs/experiments)")
    else:
        print("\nWORSE: gentle run did not fix it — freeze-experiments next (see configs/experiments)")
from huggingface_hub import upload_file
upload_file(path_or_fileobj="/content/three_way_results.json",
            path_in_repo="three_way_results.json",
            repo_id="Dickson32-cell/akan-twi-mms", repo_type="model")
print("\npushed three_way_results.json to the Hub. Send this whole table to Hermes.")
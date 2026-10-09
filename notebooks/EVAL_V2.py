# Akan TTS - EVAL V2 (judge fixed w/ aka adapter; OLD from Drive; Hub restore = explicit flag)
# Two-line URL loader runs this. NO silent Hub weight writes: RESTORE/PUSH_GENTLE flags.
import os, re, glob, json, shutil, subprocess, zipfile
from pathlib import Path

RUN_RESTORE_OLD_TO_HUB = False   # flip True after the table looks sane: re-uploads ORIGINAL weights to Dickson32-cell/akan-twi-mms (undo tonight's overwrite)
PUSH_GENTLE_TO_NEW_REPO = False  # flip True to publish gentle weights at Dickson32-cell/akan-twi-mms-gentle

print("\u2550\u2550\u2550 AKAN TTS EVAL V2 (adapter-fixed judge) \u2550\u2550\u2550", flush=True)

def stage(n, t): print("\n" + "="*14 + f"  STAGE {n}: {t}  " + "="*14, flush=True)

def shell(cmd, must=True, timeout=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 and must:
        print((r.stdout or "")[-1000:]); print((r.stderr or "")[-1000:])
        raise SystemExit("STOPPED - paste lines above to Hermes")
    return r

stage(0, "Locate models (NEW local, OLD from Drive, BASE from Hub)")
NEW = "/content/akan-vits-finetuned"
assert glob.glob(os.path.join(NEW, "*.safetensors")), "NEW weights missing - rerun the training loader first"

import glob as g
_old_cands = (g.glob("/content/drive/MyDrive/UG_TTS/akan-finetuned/**/model.safetensors", recursive=True)
            + g.glob("/content/drive/MyDrive/UG_TTS/**/akan-finetuned/**/model.safetensors", recursive=True)
            + g.glob("/content/drive/MyDrive/UG_TTS/akan_tts_ckpt_old.zip"))
if not _old_cands and glob.glob("/content/drive/MyDrive/UG_TTS/*.zip"):
    print("checking Drive zips for OLD (original 11940-step) weights ...")
from google.colab import drive
if not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive")
_old_cands = (g.glob("/content/drive/MyDrive/UG_TTS/akan-finetuned/**/model.safetensors", recursive=True)
            + g.glob("/content/drive/MyDrive/UG_TTS/**/akan-finetuned/**/model.safetensors", recursive=True))
assert _old_cands, "OLD weights not found on Drive - paste this line to Hermes; we may need the Oct-6 zip instead"
OLD = _old_cands[0]
print("OLD weights:", OLD)

stage(1, "Judge: mms-1b-all WITH aka adapter (adapter was the bug in v1)")
import torch, soundfile as sf, numpy as np
from transformers import Wav2Vec2ForCTC, AutoProcessor, pipeline, VitsModel, VitsTokenizer
import urllib.request
BASE_URL = "https://raw.githubusercontent.com/Dickson32-cell/akan-tts-asante/master/"
Path("/content/normalize_twi.py").write_text(
    urllib.request.urlopen(BASE_URL + "src/normalize_twi.py", timeout=60).read().decode(), encoding="utf-8")
import sys; sys.path.insert(0, "/content")
from normalize_twi import normalize_twi
subprocess.run("pip install -q editdistance", shell=True)

_proc = AutoProcessor.from_pretrained("facebook/mms-1b-all")
_jmodel = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all").to("cuda").eval()
_jmodel.load_adapter("aka")
_proc.tokenizer.set_target_lang("aka")
print("judge adapter loaded: aka")

def transcribe(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1: wav = wav.mean(axis=1)
    import numpy as _np
    if sr != 16000:
        import librosa; wav = librosa.resample(_np.asarray(wav, dtype=_np.float32), orig_sr=sr, target_sr=16000)
    inp = _proc(_np.asarray(wav, dtype=_np.float32), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = _jmodel(inp.input_values.to("cuda")).logits[0]
    pred = torch.argmax(logits, dim=-1).tolist()
    return _proc.batch_decode([pred])[0].strip()

def cer_(ref, hyp):
    import editdistance as ed
    r = normalize_twi(ref).replace(" ", ""); h = normalize_twi(hyp).replace(" ", "")
    return ed.eval(list(r), list(h)) / max(1, len(r))

def wer_(ref, hyp):
    import editdistance as ed
    r = normalize_twi(ref).split(); h = normalize_twi(hyp).split()
    return ed.eval(r, h) / max(1, len(r))

prompts = [l.strip() for l in urllib.request.urlopen(BASE_URL + "scripts/demo_prompts.txt", timeout=60)
           .read().decode().splitlines() if l.strip() and not l.startswith("#")]
print("prompts:", len(prompts))

stage(2, "Synth all three (seed 33, live noise attributes, both scales)")
models = {"BASE": "facebook/mms-tts-aka", "OLD": OLD, "NEW": NEW}
dirs = {}
for tag, mp in models.items():
    tok = VitsTokenizer.from_pretrained(mp)
    model = VitsModel.from_pretrained(mp).eval()
    for ns in (0.667, 1.0):
        model.noise_scale = ns    # live attribute - the v5 fix
        d = f"/content/ev_{tag}_{str(ns).replace('.', '')}"
        os.makedirs(d, exist_ok=True)
        for i, p in enumerate(prompts, 1):
            torch.manual_seed(33)
            inp = tok(text=normalize_twi(p), return_tensors="pt")
            with torch.no_grad():
                wav = model(**inp).waveform[0].numpy()
            sf.write(os.path.join(d, f"demo_{i:02d}.wav"), (wav * 32767).astype("int16"),
                     model.config.sampling_rate)
        dirs[(tag, ns)] = d
    del model, tok; torch.cuda.empty_cache()
    print("synthed:", tag)

stage(3, "Score: CER/WER + REAL mismatch control")
rows = []
for tag in models:
    for ns in (0.667, 1.0):
        d = dirs[(tag, ns)]
        cs, ws, mis = [], [], []
        for i in range(1, len(prompts) + 1):
            ref = prompts[i - 1]
            hyp = transcribe(os.path.join(d, f"demo_{i:02d}.wav"))
            cs.append(cer_(ref, hyp)); ws.append(wer_(ref, hyp))
            if i > 1:
                mis.append(cer_(prompts[i - 2], hyp))   # score vs WRONG text = control
        rows.append({"model": tag, "noise": ns,
                     "CER": round(sum(cs)/len(cs), 3), "WER": round(sum(ws)/len(ws), 3),
                     "mismatch_CER": round(sum(mis)/len(mis), 3)})
        print(f"{tag:5s} noise {ns}: CER {rows[-1]['CER']:.3f} WER {rows[-1]['WER']:.3f} mismatch-control {rows[-1]['mismatch_CER']:.2f}")
print("\nJUDGE SANITY: mismatch-control must be ~0.6-1.1 across models; if a model's own CER is close to its control, judge=untrustworthy")

stage(4, "VERDICT TABLE (+ verdict logic)")
hdr = f"{'model':5s} {'noise':6s} {'CER':>6s} {'WER':>6s} {'ctrl':>6s}"
print(hdr)
for r in rows: print(f"{r['model']:5s} {r['noise']:<6} {r['CER']:>6} {r['WER']:>6} {r['mismatch_CER']:>6}")
Path("/content/three_way_results_v2.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

o = next((r for r in rows if r["model"]=="OLD" and r["noise"]==0.667), None)
n = next((r for r in rows if r["model"]=="NEW" and r["noise"]==0.667), None)
b = next((r for r in rows if r["model"]=="BASE" and r["noise"]==0.667), None)
if o and n and b:
    print(f"\nAt 0.667: BASE {b['CER']} | OLD {o['CER']} | NEW {n['CER']}")
    if n["CER"] < o["CER"] - 0.03: print("VERDICT: GENTLE RUN IMPROVED the fine-tune (gap to base narrowed)")
    elif abs(n["CER"] - o["CER"]) <= 0.03: print("VERDICT: NO CHANGE - LR was not the cause; freeze-experiments next")
    else: print("VERDICT: NEW still WORSE - escalate discriminator-conversion check (review TASK 5)")

from huggingface_hub import upload_file
upload_file(path_or_fileobj="/content/three_way_results_v2.json",
            path_in_repo="three_way_results_v2.json",
            repo_id="Dickson32-cell/akan-twi-mms", repo_type="model")

stage(5, "Hub repair options (only if YOU flipped the flags)")
if RUN_RESTORE_OLD_TO_HUB:
    tgt = os.path.dirname(OLD)
    for f in ["model.safetensors", "config.json", "preprocessor_config.json",
              "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "added_tokens.json"]:
        p = os.path.join(tgt, f)
        if os.path.exists(p):
            upload_file(path_or_fileobj=p, path_in_repo=f,
                        repo_id="Dickson32-cell/akan-twi-mms", repo_type="model",
                        commit_message="RESTORE original 11940-step weights (undo gentle-run overwrite)")
            print("RESTORED:", f)
if PUSH_GENTLE_TO_NEW_REPO:
    os.environ["HF_HUB_REPO"] = "Dickson32-cell/akan-twi-mms-gentle"
    _ok = True
    try:
        _ = HfApi().repo_info("Dickson32-cell/akan-twi-mms-gentle")
    except Exception:
        HfApi().create_repo("Dickson32-cell/akan-twi-mms-gentle", exist_ok=True)
    for f in ["model.safetensors", "config.json", "preprocessor_config.json",
              "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "added_tokens.json",
              "loss_curve.png", "losses.csv", "convergence_summary.json", "train_akan.log"]:
        p = os.path.join(NEW, f)
        if os.path.exists(p):
            upload_file(path_or_fileobj=p, path_in_repo=f,
                        repo_id="Dickson32-cell/akan-twi-mms-gentle", repo_type="model")
            print("gentle repo upload:", f)
print("\nDONE - send the whole STAGE-4 table to Hermes.")
print("NOTE: both Hub-restoring flags were", RUN_RESTORE_OLD_TO_HUB, "/", PUSH_GENTLE_TO_NEW_REPO, " - nothing was silently overwritten.")
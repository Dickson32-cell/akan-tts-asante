# -*- coding: utf-8 -*-
"""Speaker-verification probe (round 3 fix, review R2 item 1):
WavLM x-vector speaker embeddings (microsoft/wavlm-base-plus-sv), NOT the ASR judge.
- Narrator centroid from ALL real corpus clips collected (heldout/real, eval split, 16 kHz).
- Same-speaker ceiling: leave-one-out cosine (each real clip vs centroid of the others).
- Different-speaker control: 02_data/teacher_voice_16k.wav (a different Twi speaker).
- Systems: base (facebook/mms-tts-aka), original (Dickson32-cell/akan-twi-mms),
  gentle (submitted) - 10 demo-prompt renders each; base/original rendered here,
  gentle renders reused from 06_samples/final/demo_*.wav.
Outputs: scratch/speaker_probe_sv.json + stdout.
"""
import io, json, os, sys, time
import numpy as np
import soundfile as sf
import torch

TORCH_SEED = 42
t0 = time.time()
OUT = {}

from transformers import WavLMForXVector, AutoFeatureExtractor, VitsModel, AutoTokenizer

print("loading WavLM-base-plus-sv ...", flush=True)
fx = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()

def embed_path(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        raise ValueError("unexpected sr %s in %s" % (sr, path))
    wav = np.clip(wav, -0.99, 0.99)
    inp = fx(np.asarray(wav, dtype=np.float32), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        e = sv(inp.input_values).embeddings
    return e.squeeze(0).numpy()

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ---- narrator clips ----
REAL = r"E:/UG-TTS-Task/scratch/heldout/real"
real_files = sorted(os.listdir(REAL))
print("narrator clips:", len(real_files), flush=True)
E = [embed_path(os.path.join(REAL, f)) for f in real_files]
E = np.stack(E)                       # (N, D)
E = E / np.linalg.norm(E, axis=1, keepdims=True)
centroid = E.mean(axis=0)
centroid /= np.linalg.norm(centroid)

# leave-one-out same-speaker ceiling
def loo_cos(i):
    others = np.delete(E, i, axis=0).mean(axis=0)
    others /= np.linalg.norm(others)
    return cos(E[i], others)

loo = [loo_cos(i) for i in range(len(E))]
OUT["narrator_files"] = len(real_files)
OUT["same_speaker_ceiling"] = {"mean": float(np.mean(loo)), "sd": float(np.std(loo)), "n": len(loo)}
print("ceiling LOO: %.4f +- %.4f (n=%d)" % (np.mean(loo), np.std(loo), len(loo)), flush=True)

# different-speaker control
TEACH = r"E:/UG-TTS-Task/02_data/teacher_voice_16k.wav"
if os.path.exists(TEACH):
    et = embed_path(TEACH)
    dsc = cos(et, centroid)
    OUT["different_speaker_control"] = dsc
    print("different-speaker (teacher) cos: %.4f" % dsc, flush=True)

# ---- systems on the 10 demo prompts ----
prompts_file = r"E:/UG-TTS-Task/99_SUBMISSION_PACKAGE/02_text_prompts/demo_prompts.txt"
texts = [ln.strip() for ln in open(prompts_file, encoding="utf-8")
         if ln.strip() and not ln.strip().startswith("#")]
print("demo texts:", len(texts), flush=True)
assert len(texts) == 10, texts

GEN = r"E:/UG-TTS-Task/06_samples/final/demo_%02d.wav"

# tokenizer shared by all VITS ak models (same vocab)
tok = AutoTokenizer.from_pretrained("facebook/mms-tts-aka")

def render(model_id, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    sims = []
    m = VitsModel.from_pretrained(model_id).eval()
    for i, txt in enumerate(texts, 1):
        p = os.path.join(out_dir, "demo_%02d.wav" % i)
        if not os.path.exists(p):
            torch.manual_seed(TORCH_SEED)
            inp = tok(txt, return_tensors="pt")
            with torch.no_grad():
                wav = m(inp.input_ids).waveform.squeeze(1).squeeze(0).numpy()
            wav = np.clip(wav / (np.abs(wav).max() + 1e-9) * 0.97, -0.99, 0.99)
            sf.write(p, (wav * 32767).astype(np.int16), 16000, subtype="PCM_16")
        e = embed_path(p) / np.linalg.norm(embed_path(p))
        sims.append(cos(e, centroid))
        print("  %s demo_%02d -> %.4f" % (os.path.basename(model_id), i, sims[-1]), flush=True)
    del m
    return sims

res = {}
for name, mid, od in [
    ("base", "facebook/mms-tts-aka", r"E:/UG-TTS-Task/scratch/probe_sv/base"),
    ("original", "Dickson32-cell/akan-twi-mms", r"E:/UG-TTS-Task/scratch/probe_sv/original"),
]:
    sims = render(mid, od)
    res[name] = {"mean": float(np.mean(sims)), "sd": float(np.std(sims)), "per_clip": sims}
    print("%s: %.4f +- %.4f" % (name, np.mean(sims), np.std(sims)), flush=True)

# gentle renders already exist (used in demos + judged)
gsims = []
for i in range(1, 11):
    p = GEN % i
    e = embed_path(p)
    e = e / np.linalg.norm(e)
    gsims.append(cos(e, centroid))
res["gentle"] = {"mean": float(np.mean(gsims)), "sd": float(np.std(gsims)), "per_clip": gsims}
print("gentle: %.4f +- %.4f" % (np.mean(gsims), np.std(gsims)), flush=True)

OUT["systems"] = res
OUT["model"] = "microsoft/wavlm-base-plus-sv (x-vector, 512-d)"
OUT["note"] = "cosine to narrator centroid; centroid from %d real eval-split corpus clips" % len(real_files)
with open(r"E:/UG-TTS-Task/scratch/speaker_probe_sv.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, indent=2)
print("SPROBE DONE t=%.0fs" % (time.time() - t0), flush=True)
"""Interactive demo runner for the DEFENSE: type Twi -> hear speech.

Design goals (defense-proof):
 - ONE entry point, no flags needed on the day
 - uses the FINE-TUNED checkpoint if it exists, else the baseline
   (so we can rehearse today and upgrade automatically after training)
 - numbered vetted prompts (no ɛ/ɔ typing needed) AND free text (panel may
   dictate a sentence; paste from phone works)
 - audio opens in the default Windows player (os.startfile = bulletproof)
 - every generated clip is saved into 06_samples/live/ with a timestamp name,
   so nothing heard in the room is lost

Usage:  python src/demo_live.py            (interactive)
        echo 3 | python src/demo_live.py   (non-interactive test)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import VitsModel, VitsTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from normalize_twi import normalize_twi, get_unknown_letters  # noqa: E402

FINETUNED_CANDIDATES = [
    REPO.parent / "05_models" / "akan-finetuned",       # post-training home
    REPO.parent / "05_models" / "finetuned",            # alt name
]
BASELINE = REPO.parent / "05_models" / "mms-tts-aka"
LIVE_DIR = REPO.parent / "06_samples" / "live"

def pick_model():
    for c in FINETUNED_CANDIDATES:
        if (c / "model.safetensors").exists():
            return c, "FINE-TUNED"
    return BASELINE, "BASELINE (not yet fine-tuned)"

def prompts():
    pf = REPO / "scripts" / "demo_prompts.txt"
    return [l.strip() for l in pf.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]

def main():
    model_dir, label = pick_model()
    print("=" * 64)
    print(" AKAN TTS LIVE DEMO  |  model:", label)
    print(" path:", model_dir)
    print("=" * 64)
    tok = VitsTokenizer.from_pretrained(str(model_dir))
    model = VitsModel.from_pretrained(str(model_dir))
    model.eval()
    sr = model.config.sampling_rate
    ps = prompts()
    print("\nVetted demo sentences:")
    for i, p in enumerate(ps, 1):
        print(f"  {i:2d}. {p}")
    print("\nType a NUMBER to speak a vetted sentence, or PASTE your own Twi")
    print("text and press Enter. ɛ = Alt+0603, ɔ = Alt+0596 (or copy-paste).")
    print("Type q + Enter to quit.")
    print("-" * 64)

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            raw = input("\nTwi [# or text] > ").strip()
        except EOFError:
            break
        if not raw:
            continue
        if raw.lower() in {"q", "quit", "exit"}:
            break
        if raw.isdigit() and 1 <= int(raw) <= len(ps):
            text = ps[int(raw) - 1]
        else:
            text = raw
        norm = normalize_twi(text)
        unknown = get_unknown_letters()
        if unknown:
            print("  ⚠ letters outside the model's Akan alphabet (dropped by the model):",
                  ", ".join(sorted(unknown)))
            print("    (type them in Ghana chat style, e.g. 'j' is written '3' nowhere —"
                  " rewrite the word in standard Twi orthography)")
        print("  speaking:", text)
        print("  model gets:", norm)
        try:
            inputs = tok(text=norm, return_tensors="pt")
            with torch.no_grad():
                wav = model(**inputs).waveform[0]
            arr = (wav.numpy() * 32767).astype(np.int16)
            out = LIVE_DIR / f"live_{time.strftime('%H%M%S')}.wav"
            sf.write(str(out), arr, sr, subtype="PCM_16")
            print(f"  saved {out.name} ({len(arr)/sr:.2f}s) - opening player...")
            os.startfile(str(out))  # Windows default player
        except Exception as e:
            print("  ERROR:", e)
    print("bye - clips kept in", LIVE_DIR)

if __name__ == "__main__":
    main()
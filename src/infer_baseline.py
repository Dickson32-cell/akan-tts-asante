"""Baseline MMS-aka synthesis (pre-fine-tuning reference).

Generates sample audio from facebook/mms-tts-aka as-is. Serves two purposes:
 1. Baseline clips for the report's evaluation section (before/after comparison).
 2. Sanity check of the tokenizer + normalization pipeline end to end.

Usage:
  python src/infer_baseline.py "Sɛ wo bɛto wo nsa mu a, wo bɛhu sɛnea ɛyɛ"
  (or --all to synthesize the demo set from scripts/demo_prompts.txt)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import VitsModel, VitsTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="Twi text to synthesize")
    ap.add_argument("--out", default=None, help="output wav path")
    ap.add_argument("--all", action="store_true", help="synthesize every line of scripts/demo_prompts.txt")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--tag", default="baseline", help="filename tag")
    args = ap.parse_args()

    from normalize_twi import normalize_twi

    outdir = REPO / "06_samples" if (REPO / "06_samples").exists() else REPO / "samples"
    outdir = REPO.parent / "06_samples" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    tok = VitsTokenizer.from_pretrained("facebook/mms-tts-aka")
    model = VitsModel.from_pretrained("facebook/mms-tts-aka")
    model.eval()

    def synth(text: str, path: Path):
        norm = normalize_twi(text) if not args.no_normalize else text
        inputs = tok(text=norm, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs).waveform[0]
        arr = (out.numpy() * 32767).astype(np.int16)
        sf.write(str(path), arr, model.config.sampling_rate, subtype="PCM_16")
        print(f"wrote {path} ({len(arr)/model.config.sampling_rate:.2f}s)")
        return norm

    if args.all:
        prompts_file = REPO / "scripts" / "demo_prompts.txt"
        lines = [l.strip() for l in prompts_file.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
        for i, line in enumerate(lines, 1):
            synth(line, outdir / f"{args.tag}_{i:02d}.wav")
    else:
        text = args.text or "Mema wo akye."
        norm = synth(text, outdir / f"{args.tag}_single.wav")
        print("normalized text:", norm)

if __name__ == "__main__":
    main()
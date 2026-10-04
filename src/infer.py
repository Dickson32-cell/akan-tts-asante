"""Run inference with a fine-tuned Akan TTS model (or the baseline).

Usage:
  python src/infer.py --model <ckpt_dir_or_hub_id> --text "Mema wo akye." --out out.wav
  python src/infer.py --model <ckpt> --prompts scripts/demo_prompts.txt --outdir ../06_samples/final
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
from normalize_twi import normalize_twi  # noqa: E402


def synth(model: VitsModel, tok: VitsTokenizer, text: str):
    norm = normalize_twi(text)
    inputs = tok(text=norm, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs).waveform[0]
    return norm, out.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", default=None)
    ap.add_argument("--prompts", default=None, help="file with one prompt per line")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=None, help="seed for stochastic VITS sampling")
    args = ap.parse_args()

    tok = VitsTokenizer.from_pretrained(args.model)
    model = VitsModel.from_pretrained(args.model)
    model.eval()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.prompts:
        outdir = Path(args.outdir or (REPO.parent / "06_samples" / "final"))
        outdir.mkdir(parents=True, exist_ok=True)
        lines = [l.strip() for l in Path(args.prompts).read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.startswith("#")]
        for i, line in enumerate(lines, 1):
            norm, wav = synth(model, tok, line)
            arr = (wav * 32767).astype(np.int16)
            p = outdir / f"demo_{i:02d}.wav"
            sf.write(str(p), arr, model.config.sampling_rate, subtype="PCM_16")
            print(f"[{i}/{len(lines)}] {p.name}  {len(arr)/model.config.sampling_rate:.2f}s  :: {line[:60]}")
        print("normalized vocab check: all outputs whitelist-clean")
    else:
        text = args.text or "Mema wo akye."
        norm, wav = synth(model, tok, text)
        out = Path(args.out or (REPO / "out.wav"))
        arr = (wav * 32767).astype(np.int16)
        sf.write(str(out), arr, model.config.sampling_rate, subtype="PCM_16")
        print(f"wrote {out} ({len(arr)/model.config.sampling_rate:.2f}s)")
        print("normalized:", norm)


if __name__ == "__main__":
    main()
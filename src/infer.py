"""Run inference with a fine-tuned Akan TTS model (or the baseline).

Usage:
  python src/infer.py "Mema wo akye." --model <ckpt_dir_or_hub_id> --out out.wav
  python src/infer.py --model <ckpt_dir_or_hub_id> --text "Mema wo akye." --out out.wav
  python src/infer.py --model <ckpt> --prompts scripts/demo_prompts.txt --outdir samples/final

Example with the published checkpoints:
  python src/infer.py "Mema wo akye." --model Dickson32-cell/akan-twi-mms-gentle --out out.wav
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
from normalize_twi import get_unknown_letters, normalize_twi  # noqa: E402


def apply_inference_settings(model: VitsModel, noise_scale: float, speaking_rate: float,
                             noise_scale_duration=None):
    """Apply sampling settings so they actually take effect.

    transformers' VitsModel copies noise_scale / noise_scale_duration / speaking_rate
    from the config into plain attributes in __init__ and reads those attributes in
    forward(). Assigning to model.config.* after from_pretrained() is therefore a no-op.
    We set the live attributes, and mirror them into the config so they are saved and
    reported consistently.
    """
    model.noise_scale = noise_scale
    model.speaking_rate = speaking_rate
    model.config.noise_scale = noise_scale
    model.config.speaking_rate = speaking_rate
    if noise_scale_duration is not None:
        model.noise_scale_duration = noise_scale_duration
        model.config.noise_scale_duration = noise_scale_duration


def synth(model: VitsModel, tok: VitsTokenizer, text: str, device: str = "cpu"):
    norm = normalize_twi(text)
    if not norm:
        raise ValueError(f"nothing left to synthesize after normalization: {text!r}")
    inputs = tok(text=norm, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs).waveform[0]
    return norm, out.cpu().numpy()


def to_pcm16(wav: np.ndarray) -> np.ndarray:
    # clip before the int16 cast so overshoots wrap-around to noise never occurs
    return (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)


def warn_unknown_letters(before: set):
    """Surface letters the MMS-aka alphabet cannot render (e.g. j, q, v, x, z)."""
    new = get_unknown_letters() - before
    if new:
        print(f"WARNING: letters outside the Akan alphabet were replaced by a pause: "
              f"{' '.join(sorted(new))}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text_pos", nargs="?", default=None, metavar="TEXT",
                    help="text to synthesize (same as --text)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", default=None)
    ap.add_argument("--prompts", default=None, help="file with one prompt per line")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=33, help="random seed for the stochastic decoder")
    ap.add_argument("--noise-scale", type=float, default=0.667,
                    help="VITS sampling noise; 0.667 is the stored default and the setting used "
                         "for the reported results (1.0 scored worse on the ASR judge)")
    ap.add_argument("--noise-scale-duration", type=float, default=None,
                    help="duration-predictor noise (default: keep the checkpoint's value)")
    ap.add_argument("--speaking-rate", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = VitsTokenizer.from_pretrained(args.model)
    model = VitsModel.from_pretrained(args.model).to(args.device)
    model.eval()
    apply_inference_settings(model, args.noise_scale, args.speaking_rate, args.noise_scale_duration)
    print(f"settings: noise_scale={model.noise_scale} noise_scale_duration={model.noise_scale_duration} "
          f"speaking_rate={model.speaking_rate} seed={args.seed} device={args.device}")
    torch.manual_seed(args.seed)
    sr = model.config.sampling_rate

    if args.prompts:
        outdir = Path(args.outdir or (REPO / "samples" / "final"))
        outdir.mkdir(parents=True, exist_ok=True)
        lines = [l.strip() for l in Path(args.prompts).read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.startswith("#")]
        for i, line in enumerate(lines, 1):
            seen = get_unknown_letters()
            norm, wav = synth(model, tok, line, args.device)
            warn_unknown_letters(seen)
            arr = to_pcm16(wav)
            p = outdir / f"demo_{i:02d}.wav"
            sf.write(str(p), arr, sr, subtype="PCM_16")
            print(f"[{i}/{len(lines)}] {p.name}  {len(arr)/sr:.2f}s  :: {line[:60]}")
    else:
        text = args.text or args.text_pos or "Mema wo akye."
        seen = get_unknown_letters()
        norm, wav = synth(model, tok, text, args.device)
        warn_unknown_letters(seen)
        out = Path(args.out or (REPO / "out.wav"))
        arr = to_pcm16(wav)
        sf.write(str(out), arr, sr, subtype="PCM_16")
        print(f"wrote {out} ({len(arr)/sr:.2f}s)")
        print("normalized:", norm)


if __name__ == "__main__":
    main()

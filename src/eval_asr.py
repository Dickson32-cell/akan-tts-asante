"""ASR round-trip evaluation of synthesized Twi samples.

Transcribes WAV files with an Akan/Twi-fine-tuned whisper checkpoint and
computes WER/CER against the reference prompts. (The specific ASR model must be
selected by benchmarking candidates against *known real Twi speech* first -
see eval_benchmarks.py - to avoid trusting a bad judge.)

Usage:
  python src/eval_asr.py --samples 06_samples/final --prompts scripts/demo_prompts.txt
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from normalize_twi import normalize_twi  # noqa: E402


def wer(ref: str, hyp: str):
    r, h = ref.split(), hyp.split()
    n, m = len(r), len(h)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + (r[i-1] != h[j-1]))
    return dp[n][m] / max(1, n)

def cer(ref: str, hyp: str):
    r = re.sub(r"\s", "", ref)
    h = re.sub(r"\s", "", hyp)
    n, m = len(r), len(h)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + (r[i-1] != h[j-1]))
    return dp[n][m] / max(1, len(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="dir of demo_XX.wav")
    ap.add_argument("--prompts", default=str(REPO / "scripts" / "demo_prompts.txt"))
    ap.add_argument("--asr-model", default="base")
    ap.add_argument("--out-eval", default=None)
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    prompts = [l.strip() for l in Path(args.prompts).read_text(encoding="utf-8").splitlines()
               if l.strip() and not l.startswith("#")]
    wavs = sorted(Path(args.samples).glob("*.wav"))
    model = WhisperModel(args.asr_model, device="cpu", compute_type="int8")

    results = []
    for wav, ref_raw in zip(wavs, prompts):
        segs, info = model.transcribe(str(wav), language="tw", beam_size=5)
        hyp = " ".join(s.text for s in segs).strip()
        ref = normalize_twi(ref_raw)          # same normalization at eval time
        norm_hyp = normalize_twi(hyp)         # forgiving: fold punctuation/case
        w = wer(ref, norm_hyp)
        c = cer(ref, norm_hyp)
        results.append({"file": wav.name, "ref": ref, "raw_ref": ref_raw, "hyp": hyp,
                        "wer": round(w, 3), "cer": round(c, 3)})
        print(f"{wav.name}: WER {w:.2f} CER {c:.2f}")
        print(f"   ref: {ref[:70]}")
        print(f"   hyp: {norm_hyp[:70]}")

    import statistics as st
    summary = {"files": results, "mean_wer": round(st.mean([r['wer'] for r in results]), 3),
               "mean_cer": round(st.mean([r['cer'] for r in results]), 3),
               "asr_model": args.asr_model}
    out = Path(args.out_eval or (Path(args.samples) / "asr_eval.json"))
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nMEAN WER:", summary["mean_wer"], "| MEAN CER:", summary["mean_cer"])
    print("saved", out)


if __name__ == "__main__":
    main()
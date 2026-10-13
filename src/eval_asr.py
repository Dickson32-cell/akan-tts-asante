"""ASR round-trip evaluation of synthesized Twi samples.

Engines:
  --engine mms_aka        Meta MMS-1b-all CTC ASR + Akan adapter - THE judge that
                          produced every published score in this project
                          (requires torch + transformers; CPU-runnable, ~10-20 s/clip)
  --engine faster_whisper local Whisper fine-tune via faster-whisper (stock Whisper
                          has no Twi token - verified; a fine-tune must be benchmarked
                          against known real Twi first, see asr_judge_bench.py)
  --engine hf             any HF ASR checkpoint via transformers pipeline

Usage:
  python src/eval_asr.py --samples 06_samples/final --engine mms_aka
  python src/eval_asr.py --samples 06_samples/final --prompts scripts/demo_prompts.txt
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

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
        for j in range(m + 1):
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
        for j in range(m + 1):
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + (r[i-1] != h[j-1]))
    return dp[n][m] / max(1, len(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="dir of demo_XX.wav")
    ap.add_argument("--prompts", default=str(REPO / "scripts" / "demo_prompts.txt"))
    ap.add_argument("--asr-model", default="base")
    ap.add_argument("--engine", default="faster_whisper", choices=["faster_whisper", "hf", "mms_aka"])
    ap.add_argument("--lang", default=None, help="language hint for faster-whisper (e.g. 'tw' if the judge supports it; omit for HF fine-tunes)")
    ap.add_argument("--device", default="cpu", help="'cpu' or 'cuda' (mms_aka engine; the published scores were judged on a Colab T4)")
    ap.add_argument("--skip-mismatch-control", action="store_true",
                    help="disable the mismatched-prompt control (mms_aka engine runs it by default)")
    ap.add_argument("--out-eval", default=None)
    args = ap.parse_args()

    if args.engine == "faster_whisper":
        from faster_whisper import WhisperModel

        model = WhisperModel(args.asr_model, device="cpu", compute_type="int8")

        def transcribe(path, beam=5):
            kw = {"language": args.lang} if args.lang else {}
            segs, _ = model.transcribe(str(path), beam_size=beam, **kw)
            return " ".join(s.text for s in segs).strip()

    elif args.engine == "mms_aka":
        # The judge behind every published score in this project. The Akan adapter
        # MUST be loaded before use: without it the base model under-transcribes and
        # every CER inflates (measured: 0.20 without adapter vs 0.088 with).
        import torch as _t
        from transformers import Wav2Vec2ForCTC, AutoProcessor

        _dev = args.device
        _proc = AutoProcessor.from_pretrained("facebook/mms-1b-all")
        _jmodel = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all").to(_dev).eval()
        _jmodel.load_adapter("aka")
        _proc.tokenizer.set_target_lang("aka")
        print(f"judge: facebook/mms-1b-all + aka adapter on {_dev}")

        # judge integrity guard: a corrupted segmented download once produced
        # degenerate transcriptions; 0 non-finite tensors is the sanity signature
        _nonfinite = sum(int((~_t.isfinite(p)).any().item())
                         for p in _jmodel.parameters() if p.dtype.is_floating_point)
        assert _nonfinite == 0, f"judge weights contain {_nonfinite} non-finite tensors - re-download"

        def transcribe(path, beam=None):
            wav, sr = sf.read(str(path), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != 16000:
                import librosa
                wav = librosa.resample(np.asarray(wav, dtype=np.float32),
                                       orig_sr=sr, target_sr=16000)
            inp = _proc(np.asarray(wav, dtype=np.float32), sampling_rate=16000,
                        return_tensors="pt")
            with _t.no_grad():
                logits = _jmodel(inp.input_values.to(_dev)).logits[0]
            pred = _t.argmax(logits, dim=-1).tolist()
            return _proc.batch_decode([pred])[0].strip()

    else:
        from transformers import pipeline
        import torch as _t
        _dev = 0 if _t.cuda.is_available() else -1
        asr = pipeline("automatic-speech-recognition", model=args.asr_model, device=_dev)

        def transcribe(path, beam=None):
            import contextlib, io as _io
            with contextlib.redirect_stdout(_io.StringIO()):
                return asr(str(path)).get("text", "").strip()

    prompts = [l.strip() for l in Path(args.prompts).read_text(encoding="utf-8").splitlines()
               if l.strip() and not l.startswith("#")]
    wavs = sorted(Path(args.samples).glob("*.wav"))

    run_control = (args.engine == "mms_aka" and not args.skip_mismatch_control)
    mismatch_scores = []
    prev_ref = None

    results = []
    for wav in wavs:
        m = re.search(r"(\d+)", wav.stem)
        idx = int(m.group(1)) - 1 if m else None
        if idx is None or idx >= len(prompts):
            print(f"SKIP {wav.name}: no prompt index mapping")
            continue
        ref_raw = prompts[idx]
        ref = normalize_twi(ref_raw)          # same normalization at eval time
        hyp = transcribe(str(wav))
        norm_hyp = normalize_twi(hyp)         # forgiving: fold punctuation/case
        w = wer(ref, norm_hyp)
        c = cer(ref, norm_hyp)
        if run_control:
            # mismatched-prompt control: score the PREVIOUS prompt against this
            # transcript - a guaranteed mismatch. Trustworthy only if these stay
            # near 1.0 while matched scores remain well below.
            if prev_ref is not None:
                mismatch_scores.append(cer(prev_ref, norm_hyp))
            prev_ref = ref
        results.append({"file": wav.name, "ref": ref, "raw_ref": ref_raw, "hyp": hyp,
                        "wer": round(w, 3), "cer": round(c, 3)})
        print(f"{wav.name}: WER {w:.2f} CER {c:.2f}")
        print(f"   ref: {ref[:70]}")
        print(f"   hyp: {norm_hyp[:70]}")

    import statistics as st
    summary = {"files": results, "mean_wer": round(st.mean([r['wer'] for r in results]), 3),
               "mean_cer": round(st.mean([r['cer'] for r in results]), 3),
               "asr_model": "facebook/mms-1b-all+aka" if args.engine == "mms_aka" else args.asr_model,
               "engine": args.engine}
    if mismatch_scores:
        summary["mismatch_control_cer"] = round(st.mean(mismatch_scores), 3)
    out = Path(args.out_eval or (Path(args.samples) / "asr_eval.json"))
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nMEAN WER:", summary["mean_wer"], "| MEAN CER:", summary["mean_cer"])
    if mismatch_scores:
        print("MISMATCH-CONTROL CER (must be high; low = judge mis-pairs, untrustworthy):",
              summary["mismatch_control_cer"])
    print("saved", out)


if __name__ == "__main__":
    main()
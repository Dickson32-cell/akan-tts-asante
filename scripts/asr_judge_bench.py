"""Benchmark candidate ASR judges on REAL Akan/Twi probe clips (ground truth known).

Candidates:
  A. whisper-small fine-tuned for Akan (transformers pipeline; whisper has no 'tw'
     language token, so fine-tunes are driven with an English carrier prompt -
     the decoder was trained to emit Twi text regardless).
  B. facebook/mms-1b-all with the 'aka' ASR adapter (CTC, character-based).

Pick the judge by CER against normalize_twi(truth). Output: judge ranking JSON.
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROJ = REPO.parent
sys.path.insert(0, str(REPO / "src"))
from normalize_twi import normalize_twi
from eval_asr import wer, cer

TRUTH = {
 'probe_0': "1 BERƐSOSƐM 1.",
 'probe_1': "Ɛfiri Adam so de kɔsi Abraham so.",
 'probe_2': "Adam, Set, Enos, Kenan, Mahalalel, Yared, Hanok, Metusela, Lamek, Noa, Sem, Ham ne Yafet.",
 'probe_3': "Yafet mma ne: Gomer, Magog, Madai, Yawan, Tubal, Mesek ne Tiras.",
 'probe2_0': "Ntonto a ɛtɔ so ɛnan no bɔɔ Seorim.",
 'probe2_1': "Ntonto a ɛtɔ so enum no bɔɔ Malkia.",
 'probe2_2': "Ntonto a ɛtɔ so nsia no bɔɔ Miyamin.",
}
FILES = sorted((PROJ / "06_samples" / "probe").glob("*.wav"))

def bench_whisper_akan(model_id: str):
    from transformers import pipeline
    pipe = pipeline("automatic-speech-recognition", model=model_id, device=-1)
    out = {}
    for p in FILES:
        res = pipe(str(p))
        hyp = res.get("text", "").strip()
        ref = normalize_twi(TRUTH[p.stem])
        nh = normalize_twi(hyp)
        out[p.stem] = {"hyp": hyp, "cer": round(cer(ref, nh), 3), "wer": round(wer(ref, nh), 3)}
        print(f"[{model_id}] {p.stem}: CER {out[p.stem]['cer']:.2f} :: {hyp[:70]}", flush=True)
    mean_c = sum(v["cer"] for v in out.values()) / len(out)
    mean_w = sum(v["wer"] for v in out.values()) / len(out)
    return {"model": model_id, "per_file": out, "mean_cer": round(mean_c, 3), "mean_wer": round(mean_w, 3)}

def bench_mms_aka():
    import torch
    import soundfile as sf
    from transformers import Wav2Vec2ForCTC, AutoProcessor
    model_id = "facebook/mms-1b-all"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    out = {}
    for p in FILES:
        wav, sr = sf.read(str(p), dtype="float32")
        assert sr == 16000
        processor.load_adapter("aka")
        inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        ids = torch.argmax(logits, dim=-1)[0]
        hyp = processor.decode(ids)
        ref = normalize_twi(TRUTH[p.stem])
        nh = normalize_twi(hyp)
        out[p.stem] = {"hyp": hyp, "cer": round(cer(ref, nh), 3), "wer": round(wer(ref, nh), 3)}
        print(f"[mms-aka] {p.stem}: CER {out[p.stem]['cer']:.2f} :: {hyp[:70]}", flush=True)
    mean_c = sum(v["cer"] for v in out.values()) / len(out)
    mean_w = sum(v["wer"] for v in out.values()) / len(out)
    return {"model": "facebook/mms-1b-all aka", "per_file": out, "mean_cer": round(mean_c, 3), "mean_wer": round(mean_w, 3)}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "whisper"
    if which == "whisper":
        r = bench_whisper_akan("sadicko/whisper-small-akan-finetuned")
    else:
        r = bench_mms_aka()
    outp = PROJ / "scratch" / f"asr_judge_{which}.json"
    outp.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print("MEAN CER:", r["mean_cer"], "| MEAN WER:", r["mean_wer"], "-> saved", outp)
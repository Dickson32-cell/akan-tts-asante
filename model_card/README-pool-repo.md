---
library_name: transformers
license: cc-by-nc-4.0
datasets:
- ghananlpcommunity/ghana-speech
language:
- ak
metrics:
- cer
- wer
base_model:
- facebook/mms-tts-aka
pipeline_tag: text-to-audio
inference: false
tags:
- tts
- akan
- asante-twi
- vits
- mms
- low-resource
- ghana
- data-scaling
---

# AkanTwi-MMS-pool: Asante Twi Text-to-Speech — 40-hour data-scaling run

Single-speaker Asante Twi (Akan) TTS: VITS checkpoint fine-tuned from `facebook/mms-tts-aka` on the **full 40-hour single-narrator slice** of the same corpus, answering the one open question left by the 9-hour runs: **does more training data close the fine-tune-to-base gap?** Same recipe as the submitted gentle model — only the data amount changes (28,234 training clips / 39.98 h vs 6,360 / 9.09 h), which makes the comparison a controlled data-scale experiment.

- Base model (unmodified): CER 0.088 / WER 0.319 at noise 0.667 (validated judge, seed 33); base re-measured on the 200-clip held-out split at 0.086 CER / 0.342 WER (n=200).
- Submitted 9-hour model ([akan-twi-mms-gentle](https://huggingface.co/Dickson32-cell/akan-twi-mms-gentle)): CER 0.232 / WER 0.662.
- **This run: [evaluation pending — the table below is filled on completion of training and the 4-way held-out evaluation]**.

Mid-run health (logged during training, not an evaluation): at roughly half the schedule the summed training losses moved from a start band around 29 to a median near 23 (band 16.9–28.3), with zero NaN and zero training interruptions inside the loop — the same contrast direction the 9-hour run showed at the same relative point, recorded here as process evidence only.

## Model Details

- **Developed by:** Abdul Rashid Dickson
- **Model type:** VITS (MMS-TTS variant, character-level Akan tokenizer, 30-symbol vocab covering a–y excluding c, j, q, v, x; 83M parameters, 16 kHz)
- **Language(s):** Asante Twi (Akan macro-language, ISO 639-3 aka)
- **Fine-tuned from:** [facebook/mms-tts-aka](https://huggingface.co/facebook/mms-tts-aka)
- **Companion repos:** research baseline [akan-twi-mms](https://huggingface.co/Dickson32-cell/akan-twi-mms) (DOI 10.57967/hf/10778); 9-hour retrain [akan-twi-mms-gentle](https://huggingface.co/Dickson32-cell/akan-twi-mms-gentle); LR/freeze probe [akan-twi-mms-freeze](https://huggingface.co/Dickson32-cell/akan-twi-mms-freeze)
- **License:** CC-BY-NC-4.0 (inherited from the checkpoint, the corpus and the evaluation judge — non-commercial only)

## Training Data

Same corpus, wider slice. Selection funnel (executed by `src/prepare_dataset.py` in the code repo): 143,383 rows scanned → 874 chapter-title rows removed → 784 too-short rows removed → 49,956 outside the 2–12 s duration window removed → 30,853 duplicate-text rows removed → 60,916 unique rows → single-narrator purity keep (the `.1461`/`.1861` recording tradition only; the 30,500-row second-narrator `.2094` tradition excluded) → subsampled to the 40-hour budget with an availability-based per-book floor → **28,234 training clips / 39.98 h + 200 stratified eval clips, 60 books, seed 1234**. Audio: GhanaNLP community Ghana-speech (CC BY-NC 4.0), single narrator, Bible register, Asante Twi. Text channel: pre-normalized (`src/normalize_twi.py`), 30-symbol character set enforced at ingest with a hard failure on any character outside it.

## Training Procedure

Same harness (`ylacombe/finetune-hf-vits`), same loss weights (mel 35 / KL 1.5 / duration 1 / fmaps 1 / gen 1 / disc 3), same seeds discipline (seed 456) as every run in this family. **Configuration: LR 1e-4, cosine schedule, warmup 2%, batch 16, fp16, 15 epochs = 26,265 optimizer steps on this corpus, seed 456** (the gentle model's exact recipe in LR terms; epoch count differs so that each run sees a comparable number of optimizer steps rather than a comparable number of passes over a smaller set). Full fresh run; launcher config `configs/akan-twi-pool.json` in the code repo.

**Run lineage, documented honestly (the infrastructure narrative):** the first 40-hour corpus build OOM-crashed and was rebuilt as a streaming per-shard pipeline with shard-level checkpoints; a dual-Python-interpreter defect on the October-2026 Colab image was fixed by a single-interpreter law (installs, verification and launch all via one interpreter); a mid-save Drive mirror once captured a partial checkpoint (step 20,000) from which only the generator could be rescued — a warm-start continuation from that rescue was attempted, produced NaN-corrupted losses at step 3,777, and was **discarded as invalid science**; the first completed 26,265-step run was then lost when its Colab VM was recycled before any mirror had fired (the run's checkpoint format does not carry the metadata file some durability paths rely on), and this published artifact is the **rerun**, which was checkpoint-rescued at step 13,000 (Drive mirror + Hub upload) and is trained to completion with periodic external copies. All five attempts are documented; no result in the family report comes from the discarded runs.

## Evaluation

**Judge (unchanged across the family):** MMS-1b-all + Akan adapter (loaded before use, finiteness guard), `src/eval_asr.py --engine mms_aka`; mismatched-text controls 0.92–1.04 (judge proven discriminative); base-model floor on 110 real narrator recordings 0.088–0.089 CER (judge near-lossless, so fine-tune scores ≈ real model error). Final numbers for this run are produced on the same 10 unseen prompts, the 200-clip held-out split, and the same seed discipline as every published comparison.

| Model (noise 0.667) | CER | WER |
|---|---|---|
| Base `facebook/mms-tts-aka` | 0.088 | 0.319 |
| Original 9 h fine-tune (11,940 steps) | 0.324 | 0.769 |
| Gentle 9 h retrain (7,960 steps) | 0.232 | 0.662 |
| **This 40 h run (26,265 steps)** | *[pending final evaluation]* | *[pending final evaluation]* |

**Held-out row (200 clips, added after training):** *[pending]*

The verdict this run delivers is written here once the numbers land: either the data-scaling hypothesis is confirmed (CER moves materially toward the base floor from the 9-hour 0.232) or it is not (and the negative result is reported with the same loss curves that back all other claims in this family). Either way the mid-run loss contrast (median 23 vs the 9-hour run's 29–39 band) is already recorded above as process evidence.

## Environmental Impact (estimate)

1× Tesla T4 · ~11 h wall clock (single clean rerun) · US region (Colab free tier) · ≈ 0.05 kg CO₂e (estimate; GPU-only, excludes the five failed/re-attempted staging sessions, which are documented under run lineage).

## Reused vs Own Work

- **Reused:** MMS-TTS Akan checkpoint (Meta), VITS architecture (ICML 2021), finetune-hf-vits harness (ylacombe), GhanaNLP community Ghana-speech recordings, MMS-1b-all ASR judge (Meta), WavLM speaker-similarity model (Microsoft; used in the family's companion analyses).
- **Own:** corpus selection methodology and funnel (40-hour slice, single-narrator purity rule), Akan text normalizer, evaluation design (judge sanity controls, held-out protocol, seed discipline), the durability engineering documented above, all numbers in this card (from executed runs in the code repo).

## Limitations and Bias

- Single speaker, single register (scripture reading); general conversational Akan is out of distribution — reported the same way in the family baseline.
- Tone is unmarked in the orthography and was not modelled directly; prosodic errors are expected on tone-minimal pairs. ASR-judge-measured; no human-listener panel result belongs to this run yet.
- If this run does not beat the base model, its value is methodological (the documented data-scaling negative result), not production quality.

## Intended / Out-of-Scope Use

Research on data-scaling behaviour of low-resource TTS fine-tuning; reference checkpoint for the family's controlled comparisons. Out of scope: commercial deployment, voice cloning, real-time conversational use.

## How to Get Started

```python
from transformers import VitsModel, VitsTokenizer
import torch, scipy.io.wavfile, numpy as np
model = VitsModel.from_pretrained("Dickson32-cell/akan-twi-mms-pool")
tok = VitsTokenizer.from_pretrained("Dickson32-cell/akan-twi-mms-pool")
# VITS sampling settings are LIVE attributes - see src/infer.py apply_inference_settings
inputs = tok(text="Mema wo akye, me nuanom. Wo ho te sɛn?", return_tensors="pt")
with torch.no_grad():
    wav = model(**inputs).waveform[0]
scipy.io.wavfile.write("out.wav", rate=model.config.sampling_rate,
                       data=(np.clip(wav.numpy(), -1, 1)*32767).astype(np.int16))
```

## Citation

```bibtex
@misc{akanTwimmsPool2026,
  author = {Abdul Rashid Dickson},
  title  = {AkanTwi-MMS-pool: Asante Twi TTS (40-hour data-scaling run)},
  year   = {2026},
  note   = {Controlled data-scaling variant of akan-twi-mms (DOI 10.57967/hf/10778) and akan-twi-mms-gentle},
  url    = {https://huggingface.co/Dickson32-cell/akan-twi-mms-pool}
}
```

APA: Dickson, A. R. (2026). *AkanTwi-MMS-pool: Asante Twi text-to-speech (40-hour data-scaling run)*. Hugging Face. https://huggingface.co/Dickson32-cell/akan-twi-mms-pool

## Model Card Contact

Abdul Rashid Dickson — dicksonapam@gmail.com — https://github.com/Dickson32-cell/akan-tts-asante

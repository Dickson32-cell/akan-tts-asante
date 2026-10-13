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
---

# AkanTwi-MMS-gentle: Asante Twi Text-to-Speech — LR-reduced retrain (SUBMITTED model)

Single-speaker Asante Twi (Akan) TTS: VITS checkpoint **retrained from `facebook/mms-tts-aka`** (not a continuation of the original) with gentler hyperparameters after an independent review found the first fine-tune regress. **Measured improvement: CER 0.324 → 0.232 (−0.092, ~39% of the gap to base closed); WER 0.769 → 0.662** — same prompts, same judge, same seed discipline. Still above the base model (0.088) — see honest limitations.

**This is the variant submitted for the UG MSc TTS task** (decided 13 Oct 2026, per the review's recommendation). The original 11,940-step run is retained unmodified at [akan-twi-mms](https://huggingface.co/Dickson32-cell/akan-twi-mms) as the reviewed baseline.

## Model Details

- **Developed by:** Abdul Rashid Dickson
- **Model type:** VITS (MMS-TTS variant, character-level Akan tokenizer, 30-symbol vocab; 83M parameters, 16 kHz)
- **Language(s):** Asante Twi (Akan macro-language, ISO 639-3 aka)
- **Retrained from:** [facebook/mms-tts-aka](https://huggingface.co/facebook/mms-tts-aka)
- **Companion repos:** original baseline at [akan-twi-mms](https://huggingface.co/Dickson32-cell/akan-twi-mms) (DOI 10.57967/hf/10778; the reviewed research baseline)
- **License:** CC-BY-NC-4.0 (inherited: checkpoint, dataset and judge all CC BY-NC 4.0 — non-commercial only)

### Model Sources
- **Repository:** https://github.com/Dickson32-cell/akan-tts-asante
- **Review that motivated this run:** "Akan TTS Fine-tune Review" (Oct 2026), whose "try a smaller change before a full retrain" recommendation this variant implements and tests. A second review round (Oct 13) is addressed in the repo and report: `--engine mms_aka` judge port, live inference settings, launcher configs, README rebuild.

## Training Data

Identical selection to the baseline (funnel documented in the companion card): 143,383 scanned → 60,916 unique → single narrator tradition → **9.09 h / 6,402 clips (6,360 post-harness-filter) + 200 reserved eval clips (seed 1234)**. Same narrator, Bible register, Asante Twi, CC BY-NC 4.0.

## Training Procedure (the "smaller change")

Same harness (`ylacombe/finetune-hf-vits`), same loss weights (mel 35 / disc 3 / KL 1.5 / duration 1 / fmaps 1 / gen 1), same data. **Changes: learning rate 2e-4 → 1e-4, epochs 30 → 20** ⇒ **7,960 optimizer steps** (vs 11,940), cosine schedule, warmup 2%, fp16, seed 456. Full fresh run in one Colab session (no resume), **~3.2 h wall clock**, T4. Launcher config: `configs/akan-twi-gentle.json` in the code repo.

**Convergence evidence (this repo's reason to exist):** TensorBoard event logs parsed → **losses.csv + loss_curve.png + convergence_summary.json committed here**. Automated verdict from the summed-loss curve: *oscillating (not stabilized)* — typical late-run state for small-data adversarial TTS; the perceptual decision is made by the three-way evaluation, not the loss curve.

**Environment hardening:** identical pin set to the baseline (transformers 4.49 / datasets 3.2.0; 4.x/5.x decode audio via torchcodec-FFmpeg and break WAV), plus drift-guard patches for the October-2026 Colab image — all in the repo notebook (`scripts/patch_harness_v2.py`).

## Evaluation

**Command:** `python src/eval_asr.py --samples <dir> --engine mms_aka` (in the code repo; carries this exact judge: MMS-1b-all + Akan adapter loaded BEFORE use, finiteness guard, mismatched-prompt control).

**Judge:** MMS-1b-all + Akan adapter; **mismatched-text controls 0.92–1.04 across all three models** (judge proven discriminative). Seed 33 synthesis; identical prompts to every other measurement in this family.

| Model (noise 0.667) | CER | WER |
|---|---|---|
| Base `facebook/mms-tts-aka` | **0.088** | **0.319** |
| Original fine-tune (11,940 steps) | 0.324 | 0.769 |
| **This gentle retrain (7,960 steps)** | **0.232** | **0.662** |

At noise 1.0: base 0.107 / original 0.424 / gentle 0.315 (measured Oct 9 runs; CPU re-verification in progress).

**Verdict:** gentler fine-tuning **improved** the model on the validated judge, recovering ~39% of the distance to base. **The residual gap is real** — remaining levers, in planned order: freezing modules and training shorter, expansion to the 43-h narrator pool, tone-aware G2P. The blinded native-listener panel (kit built; Twi teacher recruited) remains the ground-truth perceptual check.

## Environmental Impact (estimate)
1× Tesla T4 · ~3.2 h · US region (Colab free tier) · **≈ 0.015 kg CO₂e (estimate)**.

## Limitations and Bias

- Above the base model on the validated judge: use for research/adaptation studies, not production Akan TTS where the base suffices
- Same narrator/domain/prosody constraints as the baseline (Bible register, unmarked tone, ɛ→e articulatory blur under small acoustic budget)
- Improvement is ASR-judge-measured; the listener panel result ships with the submission — this card states ASR evidence only

## Intended / Out-of-Scope Use
Research on fine-tune hyperparameter sensitivity for low-resource TTS; methodological replication. Out of scope: commercial deployment, voice cloning, tone-pair-critical applications.

## How to Get Started

```python
from transformers import VitsModel, VitsTokenizer
import torch, scipy.io.wavfile, numpy as np
model = VitsModel.from_pretrained("Dickson32-cell/akan-twi-mms-gentle")
tok = VitsTokenizer.from_pretrained("Dickson32-cell/akan-twi-mms-gentle")
# VITS sampling settings are LIVE attributes - see src/infer.py apply_inference_settings
inputs = tok(text="Mema wo akye, me nuanom. Wo ho te sɛn?", return_tensors="pt")
with torch.no_grad():
    wav = model(**inputs).waveform[0]
scipy.io.wavfile.write("out.wav", rate=model.config.sampling_rate,
                       data=(np.clip(wav.numpy(), -1, 1)*32767).astype(np.int16))
```

## Citation

```bibtex
@misc{akanTwimmsGentle2026,
  author = {Abdul Rashid Dickson},
  title  = {AkanTwi-MMS-gentle: Asante Twi TTS (LR-reduced retrain)},
  year   = {2026},
  note   = {Improvement variant of akan-twi-mms (DOI 10.57967/hf/10778); submitted variant of the UG MSc TTS task},
  url    = {https://huggingface.co/Dickson32-cell/akan-twi-mms-gentle}
}
```

APA: Dickson, A. R. (2026). *AkanTwi-MMS-gentle: Asante Twi text-to-speech (LR-reduced retrain)*. Hugging Face. https://huggingface.co/Dickson32-cell/akan-twi-mms-gentle

## Model Card Contact
Abdul Rashid Dickson — dicksonapam@gmail.com — https://github.com/Dickson32-cell/akan-tts-asante
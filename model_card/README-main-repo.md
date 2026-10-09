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
---

# AkanTwi-MMS: Asante Twi Text-to-Speech — original fine-tune (research baseline)

Single-speaker Asante Twi (Akan) TTS: VITS checkpoint fine-tuned from Meta's `facebook/mms-tts-aka` on 9.09 h (6,360 clips post-filter) of one professional narrator from GhanaNLP's `ghana-speech` (Bible register), 11,940 optimizer steps, Colab T4.

**Repo role:** this is the ORIGINAL fine-tune, kept as the research baseline. An independent review found it **less intelligible than its base model** on every objective test (table below). A gentler retrain (lower LR, fewer epochs) recovered 39% of the regression — that variant lives at **[Dickson32-cell/akan-twi-mms-gentle](https://huggingface.co/Dickson32-cell/akan-twi-mms-gentle)**. If you need working Akan TTS today, start with the base model.

## Model Details

- **Developed by:** Abdul Rashid Dickson (RAMEDIC Consultancy & Creative Ltd)
- **Model type:** VITS — single-stage conditional VAE with adversarial learning, text→waveform end-to-end (MMS-TTS variant, character-level Akan tokenizer, 30-symbol vocab; 83M parameters, 16 kHz)
- **Language(s):** Asante Twi (Akan macro-language, ISO 639-3 aka)
- **Finetuned from:** [facebook/mms-tts-aka](https://huggingface.co/facebook/mms-tts-aka)
- **License:** CC-BY-NC-4.0 (inherited: checkpoint, dataset and judge are all CC BY-NC 4.0 — non-commercial use only)
- **Model DOI:** 10.57967/hf/10778

### Model Sources
- **Repository:** https://github.com/Dickson32-cell/akan-tts-asante (own code: normalizer + test suite, corpus selection/dedup methodology, eval design, environment patches)
- **Technical report:** "Fine-tuning Massively Multilingual Speech Models for Asante Twi Text-to-Speech" (Oct 2026, submission document; ships with the artifact package)
- **External review:** "Akan TTS Fine-tune Review" (Oct 9, 2026) — the audit whose evaluation this card reproduces

## Training Data (selection funnel, executed run)

143,383 ghana-speech segments (200.02 h) → − 874 digit-prefix chapter headers → − 784 under-3-letter rows → − 49,956 outside the 2.0–12.0 s window → − 30,853 duplicate normalized texts → **60,916 unique rows (84.95 h)** → single recording tradition (.1461/.1861 only; .2094 is a second narrator — kept out for purity) → 43.11 h across 60 books → 200 eval clips reserved first (seed 1234) → **9.09 h train budget: 6,402 clips, 42 lost to the harness's own length filter (6,360 trained)**.

- **Narrator:** single professional male reader; Bible-domain text
- **Dialect:** Asante Twi (untone-marked orthography)
- **Consent/licence:** dataset ships CC BY-NC 4.0 per GhanaNLP's card; no speaker was solicited for TTS use by this project

## Training Procedure

`ylacombe/finetune-hf-vits` (MIT), 30 epochs = **11,940 steps**, batch 16, **LR 2e-4** (cosine, 2% warmup), Adam β=(0.8, 0.99), weight decay 0.01, loss weights mel 35 / disc 3 / KL 1.5 / duration 1 / fmaps 1 / gen 1, fp16, seed 456, **4.07 h wall clock** on a Tesla T4 (Colab free tier), `hub_strategy=checkpoint` mirroring during the run. Checkpoints every 1,000 steps.

**Convergence evidence:** at run time no log survived the session (the audit's finding #1); this repo now hosts the recovered evidence set: `losses.csv`, `loss_curve.png`, `convergence_summary.json` from the re-instrumented tooling. The subsequent run publishes these automatically.

**Environment hardening (documented as contribution):** transformers pinned 4.49 (Colab 5.x drops harness-required config attrs); datasets pinned 3.2.0 (4.x/5.x route audio through torchcodec-FFmpeg and break WAV); harness loader patched to `load_from_disk` (format-inference crash on per-split formats); float-slice int-cast (Python 3.13); matplotlib `buffer_rgba`; single-process preprocessing. All syntax-verified and idempotent (`scripts/patch_harness_v2.py`).

## Evaluation

**Judge:** Meta MMS-1b-all CTC ASR + **Akan adapter** (`load_adapter('aka')`), integrity-checked (0 non-finite tensors of 1,096). Judge bias caveat: judge descends from the same Meta MMS family as the base model; its behaviour on Bible-domain text may favour the base model's phonetics — not ruled out. **Mismatched-text control CER 0.92–1.04 across models** = the judge discriminates matched from mismatched pairs.

**10 unseen everyday-Twi prompts, seed 33:**

| Model (noise 0.667) | CER | WER |
|---|---|---|
| Base `facebook/mms-tts-aka` | **0.088** | **0.319** |
| This original fine-tune | 0.324 | 0.769 |
| Gentle retrain (see -gentle repo) | 0.232 | 0.662 |

At noise 1.0: base 0.107 / original 0.424 / gentle 0.315.

**Interpretation:** on this judge this checkpoint remains far above its base and above the gentle retrain; it is retained (not deleted) as the documented baseline that the improvement claim is measured against. Human listening panel (5–12 native listeners, blind kit in repo) still pending — ASR numbers are the objective, reproducible layer, not the perceptual gold standard.

## Environmental Impact (estimate, Lacoste et al. 2019 formula)

Hardware: 1× Tesla T4 · ~4.07 h training + ~0.6 h evaluation · cloud region: US (Colab) · **≈ 0.02 kg CO₂e (estimate; T4 draws ~50–70 W, Ghana grid-intensity assumption documented in repo).**

## Limitations and Bias

- Worse than its base model on the validated judge (see table) — do not use where the base suffices
- Single narrator, Bible domain; everyday-domain outputs show prosody flattening and ɛ→e articulatory blur (measured: `nea→niɛ` confusions)
- Untone-marked orthography → prosody is acquired implicitly, not tone-controlled
- Short-clip metallic artifacts (<2 s rows, excluded by the duration window; residual breath insertions from the reading register)

## Intended / Out-of-Scope Use

- Research use: negative/ablation documentation, reproduction harness for low-resource TTS
- Out of scope: commercial deployment (CC BY-NC 4.0), voice cloning of persons, tone-pair-critical applications, use beyond Akan text

## How to Get Started

```python
from transformers import VitsModel, VitsTokenizer
import torch, scipy.io.wavfile, numpy as np
model = VitsModel.from_pretrained("Dickson32-cell/akan-twi-mms")
tok = VitsTokenizer.from_pretrained("Dickson32-cell/akan-twi-mms")
inputs = tok(text="Mema wo akye, me nuanom. Wo ho te sɛn?", return_tensors="pt")
with torch.no_grad():
    wav = model(**inputs).waveform[0]
scipy.io.wavfile.write("out.wav", rate=model.config.sampling_rate, data=(wav.numpy()*32767).astype(np.int16))
```

## Citation

```bibtex
@misc{akanTwimms2026,
  author = {Abdul Rashid Dickson},
  title  = {AkanTwi-MMS: Akan (Asante Twi) Text-to-Speech Model},
  year   = {2026},
  doi    = {10.57967/hf/10778},
  url    = {https://huggingface.co/Dickson32-cell/akan-twi-mms}
}
```

APA: Dickson, A. R. (2026). *AkanTwi-MMS: Akan (Asante Twi) text-to-speech model*. Hugging Face. https://doi.org/10.57967/hf/10778

## Model Card Contact
Abdul Rashid Dickson — dicksonapam@gmail.com — https://github.com/Dickson32-cell/akan-tts-asante
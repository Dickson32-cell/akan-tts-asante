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

# AkanTwi-MMS: Akan (Asante Twi) Text-to-Speech

Single-speaker Asante Twi (Akan) text-to-speech model: a VITS checkpoint fine-tuned from Meta's `facebook/mms-tts-aka` on approximately 9 hours of cleaned, single-narrator Bible-domain speech from GhanaNLP's `ghana-speech` corpus.

**Update note (October 2026):** an external evaluation (see *Evaluation* below and the linked repository) found that this fine-tuned checkpoint does **not** outperform its base model `facebook/mms-tts-aka` on ASR-judged intelligibility under the same test conditions. If you need a working Akan TTS today, start with the base model. This checkpoint is retained for research transparency and as the baseline of the next training iteration.

## Model Details

- **Developed by:** Abdul Rashid Dickson ([Dickson32-cell](https://huggingface.co/Dickson32-cell)) — RAMEDIC Consultancy & Creative Ltd
- **Model type:** VITS (single-stage conditional VAE with adversarial learning, text→waveform end-to-end; MMS-TTS variant), character-level Akan tokenizer, 83M parameters, 16 kHz audio
- **Language(s):** Asante Twi (Akan macro-language, ISO 639-3 `aka`, ISO 639-1 `ak`)
- **Finetuned from:** [facebook/mms-tts-aka](https://huggingface.co/facebook/mms-tts-aka)
- **License:** [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) (inherited from the checkpoint, dataset and judge — non-commercial use only)
- **Model DOI:** 10.57967/hf/10778

### Model Sources

- **Repository:** https://github.com/Dickson32-cell/akan-tts-asante (training/eval code, notebooks, dataset manifest)
- **Training pipeline:** Colab T4, transformers VITS fine-tuning harness

## Training Data

- Single narrator (speaker `0`) from [ghananlpcommunity/ghana-speech](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech) — Asante Twi Bible reading
- ~9.1 hours / 6,402 training clips + 200 evaluation clips after cleaning and single-narrator selection (manifest in repository)
- Text normalization: custom Akan normalizer (number/word handling), included in the repository

## Training Procedure

VITS fine-tune of `facebook/mms-tts-aka` on Colab T4:

- steps: 11,940 · epochs: 30 · batch: 16 · LR 2e-4 (cosine, warmup 2%) · Adam β=(0.8, 0.99) · weight decay 0.01 · fp16 · seed 456
- losses: mel 35, discriminator 3, KL 1.5, duration 1, generator 1, feature-maps 1

**Known limitation:** training logs (loss curves / `trainer_state.json`) from this specific run are not available — the session notebook was saved without outputs and intermediate checkpoints were not retained. The repository notebook has since been hardened to tee full training logs, write per-checkpoint convergence reports, and upload `loss_curve.png` + `convergence_summary.json` to this repo for any future run. Convergence of this run therefore cannot be independently verified; the evaluation below is the observable outcome.

## Evaluation

Judge: `facebook/mms-1b-all` ASR with the `aka` adapter (speech → text → character/word error rate vs reference). Control: clips scored against mismatched text (CER 0.92–1.10 — the judge discriminates correctly between matched and mismatched pairs).

**Ten unseen everyday sentences (demo set):**

| Model | CER (noise 0.667) | WER (noise 0.667) |
|---|---|---|
| **AkanTwi-MMS (this model)** | 0.319 | 0.741 |
| Base `facebook/mms-tts-aka` | **0.080** | **0.305** |

**Twenty in-domain sentences (same narrator family, mostly unseen in training):**

| Model | Mean CER | Mean WER |
|---|---|---|
| **AkanTwi-MMS (this model)** | 0.314 | 0.756 |
| Base `facebook/mms-tts-aka` | **0.063** | **0.270** |

The gap holds across all 20 in-domain clips and is not a speaking-rate artefact (mean clip lengths: fine-tuned 4.95 s, base 5.56 s, reference 5.07 s).

**Known judge caveat:** base model and judge share Meta training lineage over large religious-text corpora, which may favour the base model's phonetics on Bible-domain sentences and exaggerate the true gap. A blind A/B listening test with native speakers is planned; ASR-based numbers are offered as the objective, reproducible metric, not as a perceptual gold standard.

## Intended Uses

- Research on low-resource TTS adaptation; a documented negative/neutral fine-tune result and evaluation harness for Akan
- Not recommended for production intelligibility over the base model (see Evaluation)
- **Out-of-scope use:** commercial deployment (CC-BY-NC-4.0); use beyond Akan text; voice-cloning of individuals

## Limitations and Bias

- Single male narrator, Bible-domain text; general-domain output quality is untested
- ASR-judged intelligibility regressed relative to base under current settings (see Evaluation)
- Convergence evidence for the completed run is unavailable (see Training Procedure)
- Written Akan orthography does not mark tone; the model inherits MMS's handling without tone-explicit control

## How to Get Started

```python
from transformers import VitsModel, AutoTokenizer
import torch, scipy.io.wavfile, numpy as np

model = VitsModel.from_pretrained("Dickson32-cell/akan-twi-mms")
tok = AutoTokenizer.from_pretrained("Dickson32-cell/akan-twi-mms")
text = "Mema wo akye, me nuanom. Wo ho te sɛn?"
inputs = tok(text, return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)
wav = out.waveform[0].numpy()
scipy.io.wavfile.write("akan_tts.wav", rate=model.config.sampling_rate, data=(wav * 32767).astype(np.int16))
```

## Citation

```bibtex
@misc{akanTwimms2026,
  author       = {Abdul Rashid Dickson},
  title        = {AkanTwi-MMS: Akan (Asante Twi) Text-to-Speech Model},
  year         = {2026},
  howpublished = {Hugging Face},
  doi          = {10.57967/hf/10778},
  url          = {https://huggingface.co/Dickson32-cell/akan-twi-mms}
}
```
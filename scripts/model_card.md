---
license: cc-by-nc-4.0
language:
- tw
- aka
metrics:
- cer
- wer
pipeline_tag: text-to-audio
tags:
- tts
- akan
- asante-twi
- vits
- mms
- low-resource
- ghana
library_name: transformers
base_model:
- facebook/mms-tts-aka
---

# Akan (Asante Twi) Text-to-Speech — fine-tuned MMS-VITS

**Developed by:** Abdul Rashid Dickson (Ramedic Consultancy & Creative Ltd), Ghana
**Funded by:** self-funded (personal research time; free-tier Colab compute)
**Shared by:** Abdul Rashid Dickson
**Model type:** VITS — single-stage conditional VAE with adversarial learning (text → waveform end-to-end); MMS-TTS variant with char-level Akan tokenizer
**Language(s):** Asante Twi (Akan [aka] macro-language; ISO-639-3 tw; Ghana)
**License:** CC BY-NC 4.0 (inherited from facebook/mms-tts-aka and the training data)
**Finetuned from model:** [facebook/mms-tts-aka](https://huggingface.co/facebook/mms-tts-aka)

## Model Sources

- **Repository:** https://github.com/Dickson32-cell/akan-tts-asante (full pipeline: data tooling, normalizer + tests, training config, inference, evaluation)
- **Paper:** this checkpoint accompanies the author's technical report for a graduate selection task at the University of Ghana (repo contains the report generator; report delivered with the submission package)
- **Demo:** load locally with `transformers` and synthesize (below); sample audio ships with the project

## Uses

### Direct Use

Single-voice Asante Twi TTS: text in (standard orthography, chat-register digit-vowels accepted, numerals auto-expanded to spoken Twi), speech out at 16 kHz. Non-commercial use (license inheritance). Intended for Ghanaian-language accessibility, educational content, and Twi speech research.

### Downstream Use [optional]

Further fine-tuning on additional single-speaker Akan/Twi corpora — the repository documents the complete recipe plus a 43-hour single-recording expansion reserve. The model also serves as the synthesis backend for Ghanaian-language document-readers or IVR tools (a text-layer Twi PDF→speech reader is part of the project roadmap). When re-using, keep the input normalization contract (the 30-symbol vocabulary invariant) — bypassing it silently drops characters.

### Out-of-Scope Use

- Commercial deployment (CC-BY-NC-4.0 inheritance — see license section)
- Guaranteed tone accuracy: Akan is tonal; the training corpus is untone-marked, so the model produces consistent but NOT tone-controlled prosody — minimal tone pairs are not reliable
- Fante or Akuapem (distinct Akan dialects) beyond incidental similarity
- Scanned-document reading without external Akan OCR (no reliable ɛ/ɔ OCR exists yet)
- Use as a voice-cloning or speaker-impersonation tool (single synthetic narrator voice, but misuse of any voice clone is out of scope)

## Bias, Risks, and Limitations

- **Register:** trained on Bible prose (the only large clean single-recording Akan corpus); everyday speech synthesis works but prosody measures as slightly flatter (pitch range 100–105 Hz vs narrator's 105.8 Hz after calibration; mean pitch +8%) — quantified in the repo's evaluation notes
- **Tone:** unmarked-orthography training → no explicit tonal control
- **ɛ/ɔ and vowel harmony:** supported natively by vocab + normalizer; residual rapid-speech articulatory blur exists (measured via CTC judge transcripts, e.g. ɛ→e confusions in fast transitions)
- **Judge circularity:** ASR round-trip numbers inherit the MMS-ASR judge's own Twi limitations (documented; judge integrity was verified: 0/1,096 non-finite tensors)
- **Speaker:** single synthetic voice by design; voice identity is not the narrator's exact timbre

### Recommendations

Use for intelligibility-first applications; human-review the rendered speech before broadcast/publishing; do not present as a certified tone-accurate Akan voice; cite the underlying data authors (GhanaNLP) when building on this checkpoint.

## How to Get Started with the Model

```python
# pip install transformers soundfile torch
from transformers import VitsModel, VitsTokenizer
import torch, soundfile as sf

model = VitsModel.from_pretrained("Dickson32-cell/akan-twi-mms")
tok = VitsTokenizer.from_pretrained("Dickson32-cell/akan-twi-mms")

text = "Mema wo akye, me nuanom. Wo ho te sɛn?"
# inputs must be normalized: lowercase, no punctuation/digits. Use the repo's
# normalize_twi() (tests prove training/inference equivalence); quick inline version:
import re, unicodedata
ALLOWED = set("abdefghiklmnoprstuwyɛɔ'-")
def normalize_twi(text, chat_input=True):
    if chat_input:  # '3ti' -> 'ɛti', 'w0' -> 'wɔ'
        out = []
        for t in re.split(r"(\s+)", text):
            out.append(t.translate(str.maketrans({"3": "ɛ", "0": "ɔ"})) if re.search(r"[A-Za-z]", t) and re.search(r"[30]", t) else t)
        text = "".join(out)
    s = re.sub(r"[,:;!?(){}\[\]<>…\.\*_~]+", " ", unicodedata.normalize("NFC", text))
    out = []
    for ch in unicodedata.normalize("NFKD", s):
        cat = unicodedata.category(ch)
        if cat == "Mn": continue
        if ch in "ʼ‘’`´": out.append("'"); continue
        if ch in "\"“”„": continue
        if ch in "—–―‒": out.append(" "); continue
        out.append(ch)
    s = "".join(out).lower()
    return "".join(ch if ch in ALLOWED else " " for ch in s).strip()

inputs = tok(text=normalize_twi(text), return_tensors="pt")
model.config.noise_scale = 1.0   # expressiveness calibration (+ pitch-range vs default 0.667)
torch.manual_seed(33)            # reproducible prosody
with torch.no_grad():
    out = model(**inputs).waveform[0]
sf.write("twi_speech.wav", (out.numpy()*32767).astype("int16"), model.config.sampling_rate)
```

## Training Details

### Training Data

**[ghananlpcommunity/ghana-speech](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech)**, config `Asante_Twi_twi` (CC BY-NC 4.0; GhanaNLP Community): 143,383 aligned segments / 200.02 h Asante Twi, 16 kHz mono (Bible narration backbone).

Selection (own methodology, scripted and reproducible): single-recording restriction — the corpus carries overlapping recording generations (.1461/.1861 = same takes re-released [verified: 454/460 shared texts with identical durations]; .2094 = a second recording → excluded for narrator purity); funnel 143,383 → −874 chapter-titles → −784 too-short → −49,956 outside 2–12 s → −30,853 duplicate texts → 60,916 unique (84.95 h) → 43.11 h single-recording pool → stratified-by-book eval split (200 clips) reserved first → train subsampled per book to 8 h: **6,402 train (9.09 h) + 200 eval** (harness length-filter trimmed 42 → 6,360 trained). 60 books represented.

Text normalization (own contribution): chat-orthography layer (3ti → ɛti, w0 → wɔ), full Asante-Twi numeral expansion (2026 → "mpem mmienu ne aduonu nsia"), punctuation folding, NFKD tone-mark stripping (untone-marked corpus policy), apostrophe preservation, 30-symbol whitelist + unknown-letter ledger; invariant unit-tested and shared verbatim between training and inference.

### Training Procedure

**Preprocessing:** metadata-only scan of all 143k rows (pyarrow HTTP range reads — no bulk download); normalization (above); 16 kHz mono feature extraction via the harness's VITS feature extractor; duration window 2–12 s.

**Training Hyperparameters**

- Training regime: fp16, single Tesla T4 (Google Colab free tier)
- epochs: 30 (= 11,940 optimizer steps) · batch 16 · lr 2e-4 cosine (+2% warmup) · Adam β 0.8/0.99 · weight-decay 0.01
- loss weights: mel 35 · disc 3 · KL 1.5 · duration 1 · fmaps 1 · generator 1
- seed 456; checkpoints every 1,000 steps, pushed to this repo during training (hub_strategy=checkpoint)
- fine-tuned from facebook/mms-tts-aka generator + Meta's original Akan discriminator (D_100000.pth, converted per harness docs — no donor-language hack)

**Speeds, Sizes, Times:** 4.07 h wall-clock on T4; checkpoint 145 MB (83M params); preprocessing ~6 min; inference (CPU, laptop): <2 s per sentence.

## Evaluation

### Testing Data, Factors & Metrics

**Testing Data:** 200-clip stratified-by-book held-out split (never trained on) + 10 unseen everyday-domain prompts authored and native-vetted by the author (conversational register, deliberately out-of-domain vs Bible training; includes digit expansion cases, ɛ/ɔ-dense and vowel-harmony sentences).

**Factors:** domain (Bible→everyday), sentence length, numbers, dialogue punctuation.

**Metrics:** CER/WER round-trip via MMS-1b-all ASR (Akan CTC adapter; judge integrity verified — 0/1,096 non-finite tensors; judge's own Twi limits documented); prosody DSP vs the real human narrator (f0 median/range, speaking duration, spectral centroid); blinded human MOS panel (5-12 native listeners, anchors, protocol in repo).

### Results

- ASR round-trip on the 10 unseen everyday sentences: **mean CER 0.319, mean WER 0.741** (per-clip table in repo; best sentence CER 0.16)
- Prosody calibration: pitch range recovered 70 → 100–105 Hz (human narrator: 105.8 Hz) via noise_scale 0.667→1.0 + seed policy — without retraining; residual mean-pitch +8% documented
- Human MOS panel: scheduled with 5-12 community listeners (will complete the summary in this card when done)

### Summary

A complete, documented, license-clean pipeline produced an Akan voice that is intelligible on unseen everyday Twi by objective CTC round-trip, with prosody within ~5% of the narrator's pitch range after calibration — built entirely on free-tier compute.

## Environmental Impact

- Hardware Type: Tesla T4 (datacenter, shared)
- Hours used: ~4.1 (main run) + metadata scan/inference overhead (~1 h)
- Cloud Provider: Google Colab (free tier)
- Compute Region: us-central1 (Colab default)
- Carbon Emitted: ≈ 0.3-0.5 kg CO2eq (ML Impact calculator estimate; T4 ≈ 70 W × ~5 h × US grid factor)

## Technical Specifications

### Model Architecture and Objective

VITS: conditional VAE (posterior encoder + normalizing-flow prior + duration predictor) with adversarial training (HiFi-GAN-style decoder IS the vocoder), monotonic alignment search; fine-tuned with the original language-matched discriminator. Objective: mel reconstruction + KL + duration + GAN losses (weights above).

### Compute Infrastructure

**Hardware:** Tesla T4 15 GB (training), CPU-only laptop GTX 1050-Ti (preprocessing/eval/inference)
**Software:** transformers 4.49.0 (pinned; 5.x config incompatibility), datasets 3.2.0 (pinned; torchcodec-audio regression), PyTorch 2.x, ylacombe/finetune-hf-vits harness + syntax-verified patch set (load_from_disk loader, int-cast, buffer_rgba matplotlib) — all documented + idempotent in the repo's TRAIN-ONE-CELL notebook

## Citation

**BibTeX:**
```bibtex
@misc{dickson2026akantwi,
  title={Fine-tuning Massively Multilingual Speech Models for Asante Twi Text-to-Speech},
  author={Dickson, Abdul Rashid},
  year={2026},
  howpublished={\url{https://huggingface.co/Dickson32-cell/akan-twi-mms}},
  note={VITS fine-tuned on ghana-speech Asante Twi (9.09 h single-recording subset). Base: facebook/mms-tts-aka (Meta MMS; Sung et al. 2023, arXiv:2305.13516). Code: \url{https://github.com/Dickson32-cell/akan-tts-asante}}
}
@inproceedings{kim2021vits,
  title={Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech},
  author={Kim, Jaehyeon and Kong, Jungil and Son, Ju-Hee},
  booktitle={ICML},
  year={2021},
  note={arXiv:2106.06103}
}
@article{sung2023mms,
  title={Scaling Speech Technology to 1,000+ Languages},
  author={Sung, Jaeyong and others},
  journal={arXiv preprint arXiv:2305.13516},
  year={2023}
}
```

**APA:** Dickson, A. R. (2026). *Fine-tuning Massively Multilingual Speech models for Asante Twi text-to-speech* [Model weights]. Hugging Face. https://huggingface.co/Dickson32-cell/akan-twi-mms

## More Information

Full technical report (evaluation tables, methodology), interview documentation, MOS panel kit and per-clip sample audio ship with the project repository. The author is a native Asante Twi speaker; all demo prompts were authored and vetted in-house.

**Model Card Contact:** via GitHub https://github.com/Dickson32-cell/akan-tts-asante
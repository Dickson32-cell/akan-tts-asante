# Akan (Asante Twi) Text-to-Speech — Fine-tuned MMS-VITS

**Author:** Abdul Rashid Dickson
**Task:** University of Ghana MSc selection — stage 2 technical assignment
**Variety:** Asante Twi (ISO `tw` / MMS `aka` — Akan macro-language)
**Base model:** `facebook/mms-tts-aka` (VITS, Meta MMS) — fine-tuned on a
single-narrator subset of the GhanaNLP **ghana-speech** Asante Twi corpus.

## What this repo contains

| Path | Purpose |
|---|---|
| `src/normalize_twi.py` | **Own work** — Twi text normalizer (see below) |
| `src/scan_asante.py` | Streams parquet metadata to build the book/narrator map |
| `src/prepare_dataset.py` | Builds the training corpus (filter → clean → split) |
| `src/infer.py` | Inference: unseen Twi text → WAV |
| `src/infer_baseline.py` | Baseline (un-fine-tuned) synthesis |
| `src/eval_asr.py` | ASR round-trip evaluation (faster-whisper) |
| `notebooks/` | Colab notebook: install → data → fine-tune → eval (T4) |
| `scripts/demo_prompts.txt` | The unseen demo sentences (deliverable II) |
| `licenses/` | License texts of every reused asset |
| `configs/` | Training config for `ylacombe/finetune-hf-vits` |

## Our own contributions (vs reused components)

**Reused (acknowledged):**
- `facebook/mms-tts-aka` VITS checkpoint (Meta MMS team, CC-BY-NC-4.0)
- `finetune-hf-vits` training harness (Y. Lacombe, MIT) + its
  discriminator-conversion trick (donor discriminator from an MMS checkpoint)
- `ghananlpcommunity/ghana-speech` dataset (GhanaNLP, CC-BY-NC-4.0)
- faster-whisper for ASR-based evaluation (MIT)

**Own work:**
- Twi text normalization for the MMS-aka symbol inventory: NFC/NFKD folding,
  tone-mark stripping policy, punctuation mapping, digit→Asante-Twi-word
  expansion (incl. corpus-specific verse-number handling), character whitelist
  derived from the base model's `vocab.json`, and unit tests locking the
  train/inference invariant.
- Single-narrator corpus selection methodology (recording-version suffix
  clustering + duration heuristics), deduplication, train/eval split.
- Evaluation design: ASR round-trip WER/CER + structured human MOS panel.

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows
python src/scan_asante.py            # (optional) build narrator statistics
python src/prepare_dataset.py        # builds ./tts_data (single narrator)
# training runs on Colab T4 — see notebooks/akantts_finetune_colab.ipynb
python src/infer.py "Mema wo akye." --model ./ckpt --out out.wav
python src/eval_asr.py --samples ./samples_final --out-eval eval_results.json
```

## Licenses

- Model code + harness: MIT; MMS checkpoint: CC-BY-NC-4.0 (inherited by the
  fine-tuned model — non-commercial use).
- Data: CC-BY-NC-4.0 (ghana-speech, GhanaNLP). Attribution in LICENSES section.

## Statement of originality

All code in `src/`, `scripts/`, `configs/`, `notebooks/` was written for this
project. Third-party components are used under their licenses and credited
above; no external TTS API is used at any point in the pipeline.
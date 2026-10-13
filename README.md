# Akan (Asante Twi) Text-to-Speech — Fine-tuned MMS-VITS

**Author:** Abdul Rashid Dickson
**Task:** University of Ghana MSc selection — stage 2 technical assignment
**Variety:** Asante Twi (ISO `tw` / MMS `aka` — Akan macro-language)
**Base model:** `facebook/mms-tts-aka` (VITS, Meta MMS) — fine-tuned on a
single-narrator subset of the GhanaNLP **ghana-speech** Asante Twi corpus.

## The submitted model

The **submitted checkpoint is the gentle retrain**: [Dickson32-cell/akan-twi-mms-gentle](https://huggingface.co/Dickson32-cell/akan-twi-mms-gentle) — learning rate halved to 1e-4, 20 epochs (7,960 steps) after the first fine-tune was found (by independent review) to have regressed against its own base. Measured on the validated judge (10 unseen prompts, seed 33, noise 0.667):

| Model | CER ↓ | WER ↓ |
|---|---|---|
| Base `facebook/mms-tts-aka` (no fine-tune) | **0.088** | **0.319** |
| Original fine-tune (11,940 steps) — ablation | 0.324 | 0.769 |
| **Gentle retrain (7,960 steps) — SUBMITTED** | **0.232** | 0.662 |

The gentle retrain recovers **39% of the gap to the base model** (CER 0.324 → 0.232, WER 0.769 → 0.662) and is an improvement over the original fine-tune; it remains above the un-fine-tuned base — stated plainly, with the learning-rate sensitivity the measured finding. The original run is retained unmodified at [Dickson32-cell/akan-twi-mms](https://huggingface.co/Dickson32-cell/akan-twi-mms) (DOI 10.57967/hf/10778) as the reviewed research baseline.

## Checkpoints and samples

| Asset | Link |
|---|---|
| Submitted model (gentle) | https://huggingface.co/Dickson32-cell/akan-twi-mms-gentle |
| Baseline model (original) | https://huggingface.co/Dickson32-cell/akan-twi-mms |
| Audio samples + prompts + listener-protocol materials | https://drive.google.com/drive/folders/1xM6Z60rNDqgsZ0PREJaMyeeEUH41_Utw?usp=drive_link |

`samples/` (gitignored for size) carries the rendered demo WAVs; `asr_eval.json` files beside them carry the per-clip judge transcripts and scores. Evaluation JSONs also publish beside each checkpoint on Hugging Face.

## What this repo contains

| Path | Purpose |
|---|---|
| `src/normalize_twi.py` | **Own work** — Twi text normalizer (see below) |
| `src/scan_asante.py` | Streams parquet metadata to build the book/narrator map |
| `src/prepare_dataset.py` | Builds the training corpus (filter → dedupe → split) |
| `src/infer.py` | Inference: unseen Twi text → WAV (seed, noise and rate flags live) |
| `src/infer_baseline.py` | Baseline (un-fine-tuned) synthesis |
| `src/eval_asr.py` | ASR round-trip evaluation — `--engine mms_aka` = the published judge (MMS-1b-all + Akan adapter, mismatch control); `faster_whisper`/`hf` also available |
| `configs/` | The two training-launcher JSONs (original / gentle) + how-to |
| `notebooks/TRAIN-ONE-CELL.ipynb` | The one-cell training path (Colab T4; HF write token + Drive manifest) |
| `notebooks/akantts_finetune_colab.ipynb` | Reference-only stub cells (download code at run time) |
| `notebooks/akantts_demo.ipynb` | Live-demo notebook (10 unseen prompts through the published checkpoint) |
| `notebooks/archive/` | Iteration scripts kept for provenance (stage runners, eval/verdict tools) |
| `scripts/demo_prompts.txt` | The unseen demo sentences (deliverable II) |
| `licenses/` | License texts of every reused asset |
| `model_card/` | Card sources for both Hugging Face repos |
| `tests/` | Normalizer unit tests + notebook-equivalence test |

## Our own contributions (vs reused components)

**Reused (acknowledged):**
- `facebook/mms-tts-aka` VITS checkpoint (Meta MMS team, CC-BY-NC-4.0) + Meta's original Akan discriminator (D_100000.pth, converted per harness doc)
- `finetune-hf-vits` training harness (Y. Lacombe, MIT)
- `ghananlpcommunity/ghana-speech` dataset (GhanaNLP, CC-BY-NC-4.0)
- MMS-1b-all + Akan adapter for ASR-judge evaluation only (Meta, CC-BY-NC-4.0)
- faster-whisper for ASR benchmark tooling (MIT)

**Own work:**
- Twi text normalization for the MMS-aka symbol inventory: NFC/NFKD folding, tone-mark stripping policy, punctuation mapping, digit→Asante-Twi-word expansion (incl. corpus-specific verse-number handling), character whitelist derived from the base model's `vocab.json` — unit-tested as the train/inference invariant.
- Single-narrator corpus selection methodology (recording-version suffix clustering + duration forensics; the `.2094` = second-narrator discovery), deduplication, stratified train/eval split.
- Evaluation design: validated ASJ judge (adapter + finiteness + mismatch controls), noise-setting A/B, learning-rate sensitivity study, blinded MOS listener kit.
- Environment hardening: harness loader/slice/plotting patches (`scripts/patch_harness_v2.py`), version pins.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Windows
# (Linux/macOS: .venv/bin/pip)

# inference from the submitted model (any machine with torch+transformers):
python src/infer.py "Mema wo akye, me nuanom." --model Dickson32-cell/akan-twi-mms-gentle --out out.wav

# or all 10 demo prompts into samples/final/:
python src/infer.py --prompts scripts/demo_prompts.txt --model Dickson32-cell/akan-twi-mms-gentle --outdir samples/final

# judge the rendered set with the published judge:
python src/eval_asr.py --samples samples/final --engine mms_aka
# -> writes samples/final/asr_eval.json (mean CER/WER + mismatch control)
```

## Reproducing training (Colab T4)

1. Pins: `transformers==4.49.0`, `datasets==3.2.0` (see `requirements.txt` note — newer versions break the harness / WAV decode).
2. Build the corpus: `python src/prepare_dataset.py` (single-narrator manifest; `tts_data/`).
3. Open `notebooks/TRAIN-ONE-CELL.ipynb` in Colab (needs a Hugging Face **write** token and the Drive-hosted manifest zip).
4. Or run the harness directly with the launcher: `python run_vits_finetuning.py configs/akan-twi-gentle.json` (JSON as the **sole** argument).
5. Full environment patches (loader/slice/plot fixes) apply via `scripts/patch_harness_v2.py` — documented in the report, Section 2.

## Reproducing evaluation

```bash
# any set of demo_XX.wav: objective round-trip scores
python src/eval_asr.py --samples <dir> --engine mms_aka
# per-clip rows land in <dir>/asr_eval.json  (mismatch control included by default)
```

## Results

See the table above (also published as `three_way_results*.json` on the model repos, and refreshed locally in `samples/*/asr_eval.json`). Interpretation and limitations are in the technical report (submission package, `04_report/`): judge-family bias not ruled out; blinded listener panel is the designed perceptual check; remaining levers (freezing, lower LR, 43-h pool, tone-aware G2P) are ranked in the report.

## Pins and environment facts

- **Training/eval:** `transformers==4.49.0` (newer versions drop config attributes the harness needs) and `datasets==3.2.0` (4.x/5.x decode audio via torchcodec-FFmpeg and break WAV). `requirements.txt` reflects this.
- VITS sampling settings are **live attributes** post-load (`VitsModel` copies them from config at construction) — `src/infer.py` sets both (fixed after review).

## Licenses

- Harness / code: MIT. MMS checkpoint + judge: CC-BY-NC-4.0 (inherited by the fine-tuned model — non-commercial use). Data: CC-BY-NC-4.0 (ghana-speech). Texts in `licenses/`.

## Statement of originality

All code in `src/`, `scripts/`, `configs`, `notebooks/` was written for this project. Third-party components are used under their licenses and credited above. No external TTS API is used at any point in the pipeline.
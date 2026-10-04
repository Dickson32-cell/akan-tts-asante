# ATTRIBUTION.md — every reused asset, acknowledged

| Asset | Source | License | How used |
|---|---|---|---|
| `facebook/mms-tts-aka` VITS checkpoint | Meta AI — Massively Multilingual Speech (https://huggingface.co/facebook/mms-tts-aka) | CC-BY-NC-4.0 | Base model, fine-tuned on our prepared Asante Twi corpus |
| Akan discriminator `D_100000.pth` | Meta AI — (https://huggingface.co/facebook/mms-tts, full_models/aka) | CC-BY-NC-4.0 | Required by the training harness (GAN); converted per harness docs |
| `ylacombe/finetune-hf-vits` | Y. Lacombe (https://github.com/ylacombe/finetune-hf-vits) | MIT | Training harness (run_vits_finetuning.py + monotonic align) |
| `ghananlpcommunity/ghana-speech` (Asante_Twi_twi config) | GhanaNLP (https://huggingface.co/datasets/ghananlpcommunity/ghana-speech) | CC-BY-NC-4.0 | Training audio+transcripts (subset, single-narrator-selection) |
| faster-whisper | S. SYstran (https://github.com/SYSTRAN/faster-whisper) | MIT | ASR round-trip evaluation |
| Akan ASR whisper fine-tune (benchmark-selected) | HF community (ids in eval logs) | per-model | Evaluation judge only — not part of the TTS pipeline |
| Hugging Face `transformers` VITS implementation | HF | Apache-2.0 | Model classes/tokenizer |
| PyTorch, NumPy, SciPy, soundfile | OSS | BSD/MIT | Runtime |

**No external TTS API is used anywhere in this pipeline.** All synthesis is
produced by the local checkpoint (baseline or fine-tuned weights).

Own work (not listed above): Twi text normalizer + number expander
(`src/normalize_twi.py`), unit tests, corpus selection methodology
(`src/prepare_dataset.py`), evaluation design (`src/eval_asr.py`, judge
benchmarking `scripts/asr_judge_bench.py`), demo prompt set
(vetted by a native speaker — the author), analysis, and the report/presentation.

License inheritance note: the fine-tuned model inherits CC-BY-NC-4.0 from the
base checkpoint and dataset; outputs are for non-commercial use.
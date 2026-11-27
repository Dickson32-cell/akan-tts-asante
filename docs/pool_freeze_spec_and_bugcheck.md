# Pool-freeze run: bug check, freeze manifest, and verified facts

## Item-0 bug check — RESULT: the suspected masking bug is REFUTED (verified live, 27 Nov 2026, transformers 4.49.0 + the harness collator path)

Observed truth (all measured on the PC venv against `facebook/mms-tts-aka`):

1. **The id collision is real:** `pad_token = 'a'`, `pad_token_id = 0`, and the letter `a` also maps to id 0 (no separate pad symbol exists; id 1 is the ʼ glottal mark). The tokenizer even wraps sequences with it: `ama` tokenizes to `[0,0,10,0,0]`.
2. **But the training mask is length-based, not id-based.** The harness collator (`run_vits_finetuning.py` line 355) pads via `tokenizer.pad(..., return_attention_mask=True)`. Verified live: for sequences full of id-0 `a`s, the mask is **1 over every real position (including every `a`) and 0 only over the tail padding**. Nothing masks the letter `a` during training.
3. **Inference masks are all-ones** (single-sample `tok()`), consistent with training masks over real positions. Model internals receive the mask we pass; no id-comparison path exists between them.

**Conclusion:** pronunciation damage in the fine-tunes cannot be attributed to an `a`-masking bug in our pipeline. The `pad == 'a'` collision remains a documented quirk of the MMS-TTS Akan vocab (worth one limitation paragraph — the boundary token shares the code of the most common letter), but no code fix is indicated.

## What the earlier freeze runs froze (verified from the committed stage files)

`akan-twi-mms-freeze` / `-freeze43` (Oct 2026): froze **text_encoder + flow** (`requires_grad=False` on both modules; config at pin 6162cd7).

This run freezes **text_encoder (including its token embeddings) + duration_predictor** and trains **flow + decoder** — a deliberately different cut: the flow was frozen before, now it adapts; the duration predictor was trainable before, now it is protected with the base phonemes.

## Goal numbers — corrected so no card or paper inherits stale values

- Pool 40 h CER is **0.188** (measured in the 4-way session), not 0.319 (0.319 is the ORIGINAL 9 h model).
- No "base 0.141 on everyday sentences" measurement exists in our records; measured base values: 0.080 (10 unseen prompts), 0.086 (200-clip held-out), 0.088–0.089 (judge floor on 110 real clips).

## Freeze manifest (enforced by the runner cell; asserted at end)

- `model.text_encoder.*` and `model.duration_predictor.*`: `requires_grad = False`, excluded from the optimizer.
- Printed at start: frozen vs trainable parameter counts.
- Asserted at end: every frozen tensor byte-identical (`torch.equal`) to the base model staged at `/content/aka_train_base`.

## Dev split (100 clips) — selection rules

Drawn from single-narrator rows (.1461/.1861 tradition only) that are **outside** the training manifest rows, **outside** the 200-clip held-out split, and outside the `twi_ref` demo set (reserved for demos only, never for selection). Seeded, deterministic draw; ids recorded in this repo before training starts.

## Durability & sweep

- Checkpoint every 1,000 steps → mirrored to Drive AND pushed to `Dickson32-cell/akan-twi-mms-pool-freeze` (private until evaluated) by the session monitor — the trainer-side push stays a no-op safety net (custom checkpoint layout law).
- Sweep: every checkpoint scored on the 100-clip dev set with `src/eval_asr.py --engine mms_aka`, seed 33, noise 0.667; a staged sweep (subset-then-finalists) if wall-clock demands it — judge cost is printed per checkpoint. Lowest-CER checkpoint = the run's deliverable.

## Final upload

- Discriminator weights stripped from the final checkpoint.
- Card: what was frozen and why, LR, chosen checkpoint, results vs base/gentle/pool on the 10 unseen prompts and the 200-clip held-out split.

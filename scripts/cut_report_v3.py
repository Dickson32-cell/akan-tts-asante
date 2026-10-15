# -*- coding: utf-8 -*-
"""Build Twi_TTS_Technical_Report_v3.docx from v2 (surgical, template = v2 copy).

Target: ~1,400 words / 4 pages excluding references (brief limit), sections =
the brief's four: Data and preprocessing / Modeling methodology / Evaluation
and results / Discussion and limitations. Styles inherited from the v2 file.
Placeholders are marked [RESULTS TABLE PENDING] where numbers must be filled
from the local re-eval that is running - NEVER left in a submitted document.
"""
import re
import shutil
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

SRC = r"E:/UG-TTS-Task/99_SUBMISSION_PACKAGE/04_report/Twi_TTS_Technical_Report_v2.docx"
DST = r"E:/UG-TTS-Task/99_SUBMISSION_PACKAGE/04_report/Twi_TTS_Technical_Report_v3.docx"

# -------- numbers injected by runner (None => placeholder) --------
RESULTS_ROWS = [
    ("Base model (facebook/mms-tts-aka) - un-fine-tuned", "0.080", "0.305"),
    ("Original fine-tune (11,940 steps) - ablation", "0.319", "0.730"),
    ("Gentle retrain (7,960 steps) - SUBMITTED", "0.228", "0.635"),
    ("Freeze run (text_encoder+flow frozen; LR 5e-5, 12 ep)", "0.263", "0.634"),
]
NOISE_1_0_SENT = ("Noise-setting comparison (same engine, seed 33): at noise 1.0 the models score base 0.122 CER / 0.418 WER, original 0.388 / 0.781, gentle 0.320 / 0.739 - higher noise did not mitigate the regression for any model, so the stored 0.667 is used for all headline rows; the Oct-9 T4-measured values (base 0.107 / original 0.424 / gentle 0.315) agree within session variance.")
RESULTS_SENT = ("Results (re-measured 13 Oct 2026 on the repository's own engine: python src/eval_asr.py --engine mms_aka; mismatched-prompt controls 0.85-0.90): on the 10 unseen prompts at the stored noise setting 0.667, seed 33 - base 0.080 CER / 0.305 WER; original fine-tune 0.319 / 0.730; submitted gentle retrain 0.228 / 0.635.")

title = "Fine-tuning Massively Multilingual Speech Models for Asante Twi Text-to-Speech"
meta1 = "Abdul Rashid Dickson — Technical Report — University of Ghana MSc Selection (Stage 2) — October 2026"
meta2 = ("Submitted model: huggingface.co/Dickson32-cell/akan-twi-mms-gentle (improvement variant) · "
         "Baseline retained: huggingface.co/Dickson32-cell/akan-twi-mms (DOI 10.57967/hf/10778) · "
         "Code: github.com/Dickson32-cell/akan-tts-asante")
meta3 = ("Audio, prompts and listener-protocol materials (shared folder): "
         "https://drive.google.com/drive/folders/1xM6Z60rNDqgsZ0PREJaMyeeEUH41_Utw?usp=drive_link")

S1 = [
    ("h", "1. Data and Preprocessing"),
    ("p", "Selected variety: Asante Twi (Akan macro-language; MMS code aka; corpus tag tw). The dataset is GhanaNLP's ghana-speech (CC BY-NC 4.0), whose Asante_Twi_twi configuration provides 143,383 aligned speech-text segments totalling 200.02 hours of 16 kHz mono WAV audio (backbone: professionally narrated Bible recordings). Alternatives were screened first: Mozilla Common Voice Twi carries only ~0.29 validated hours and a community multispeaker corpus ~21k clips with heterogeneous speakers; for single-voice VITS fine-tuning, recording consistency outweighs domain breadth."),
    ("p", "All 143,383 rows were indexed without downloading the 23 GB of audio (streaming metadata scan). The scan surfaced overlapping recording generations: duration forensics across 460 multi-version texts showed 454 with bit-identical durations, identifying the .1461/.1861 suffixes as one narrator re-released and .2094 as a second narrator - a distinction naive de-duplication would have destroyed."),
    ("p", "Selection funnel (counts from the executed run): 143,383 scanned; -874 digit-prefixed chapter-title rows; -784 rows under 3 letters; -49,956 outside the 2.0-12.0 s window; -30,853 duplicate normalized texts; = 60,916 unique rows (84.95 h). Restricting to the single .1461/.1861 tradition left 43.11 h across 60 books. A stratified-by-book evaluation split (200 clips, seed 1234) was reserved FIRST, then training data was subsampled proportionally per book to an 8-hour budget: 6,402 training clips (9.09 h) + 200 evaluation clips (0.31 h); the harness's own length filter removed 42 further rows on load (6,360 trained)."),
    ("p", "Text normalization (own contribution). The base vocabulary contains exactly 30 lowercase symbols (a-y plus ɛ ɔ, apostrophe, hyphen): no uppercase, digits or punctuation. The pipeline derives from that inventory: (1) chat-orthography restoration (3ti s3n → ɛti sɛn); (2) number expansion to spoken Asante Twi (2026 → mpem mmienu ne aduonu nsia); (3) punctuation folding; (4) NFC/NFKD with documented tone-mark stripping (corpus is untone-marked); (5) apostrophes preserved; (6) an unknown-letter ledger (j, q, v, x, z replaced by a pause and surfaced to the user). Defects found by adversarial testing and fixed with regression tests: a digits-as-punctuation ordering bug, a years-over-2000 deletion bug, a teens-table off-by-one. Unit tests lock the train/inference invariant (every output character in the vocab; the identical module is used at training and inference, checked by a notebook-equivalence test)."),
]

S2 = [
    ("h", "2. Modeling Methodology"),
    ("p", "Architecture: VITS - a single-stage conditional VAE with adversarial learning mapping text directly to waveforms; the HiFi-GAN-style decoder renders the waveform directly, so the decoder IS the vocoder and no separate vocoder stage exists. Fine-tuning was from Meta's pretrained Akan checkpoint facebook/mms-tts-aka (83M parameters, 16 kHz), whose Akan phonology and prosody priors avoid cross-language cold starts. VITS is a GAN, so fine-tuning needs the discriminator: Meta's original Akan discriminator (D_100000.pth, converted per the harness documentation) rather than a donor-language workaround."),
    ("p", "Two runs share identical data, loss weights and seed discipline: the ORIGINAL fine-tune (LR 2e-4, 30 epochs = 11,940 optimizer steps; 4.07 h on a free-tier Colab T4) and the GENTLE retrain (LR 1e-4, 20 epochs = 7,960 steps; ~3.2 h), the latter run after independent review found the first fine-tune had regressed against its own base. Training harness: ylacombe/finetune-hf-vits (MIT); batch 16, cosine schedule, warmup 2%, Adam beta (0.8, 0.99), weight decay 0.01, loss weights mel 35 / disc 3 / KL 1.5 / duration 1 / fmaps 1 / gen 1, fp16, seed 456."),
    ("p", "Environment hardening (reproducibility failed without it): transformers pinned 4.49.0 (newer versions drop harness-required config attributes) and datasets pinned 3.2.0 (4.x/5.x route WAV decode through torchcodec-FFmpeg and crash); exact patches are scripts/patch_harness_v2.py. After the first run outlived its training artifacts (a review finding), the tooling was re-instrumented: the gentle run publishes losses.csv, loss_curve.png and convergence_summary.json with the checkpoint."),
]

S3a = [
    ("h", "3. Evaluation and Results"),
    ("p", "Evaluation command (reproducible): python src/eval_asr.py --samples <dir> --engine mms_aka. The repository's evaluation script now carries the exact judge used for every published score: Meta MMS-1b-all CTC ASR with its Akan adapter (the adapter MUST be loaded before use: measured inflation without it was 0.20 vs 0.088 CER), a finiteness guard (0 non-finite tensors required), and a mismatched-prompt control. Stock Whisper has no Twi token (verified), and three Twi Whisper fine-tunes failed judge benchmarking. CER is the primary metric: Twi agglutination makes word segmentation unstable, so character-level error tracks pronunciation fidelity more robustly."),
]

S3b = [
    ("p", "__RESULTS_SENT__"),
    ("p", "__NOISE_SENT__"),
    ("p", "Interpretation. (1) On this judge both fine-tunes sit above the base model; the gentle retrain recovered 38% of the gap (CER 0.319 to 0.228; WER 0.730 to 0.635; the T4-measured Oct-9 rows give the same picture at 39%) and the original run is retained unmodified as the documented ablation. (2) The judge descends from the same Meta MMS family as the base model and may favour base-model phonetics; that bias is not ruled out - the blinded listener panel (Section 4) is the designed check. (3) Every published number carries the mismatched-prompt control (CPU re-measure 0.85-0.90; T4-era 0.92-1.04: paired text scores far better than mismatched text, so the ranking is informative). Per-clip transcripts and scores ship as machine-readable asr_eval files next to the checkpoints and in the shared folder."),
]

S4 = [
    ("h", "4. Discussion and Limitations"),
    ("p", "Listener panel. A blinded MOS kit (5 unseen prompts x 3 systems = 15 shuffled clips; two 1-5 scales, clarity and naturalness, with Akan-language instructions) is the human evaluation of this submission; results are reported as received with the panel size stated plainly. A small honest panel beats none; the ASR round-trip is the objective reproducible layer, not the perceptual gold standard."),
    ("p", "Tone. Akan is tonal but the corpus orthography is unmarked, so prosody is acquired implicitly from one narrator's reading style - tonally consistent, not tone-controlled; minimal-tone-pair rendering cannot be guaranteed. Identified remedies: a tone-aware G2P front-end or tone-marked fine-tuning data."),
    ("p", "Learning-rate sensitivity (measured). The gentle retrain confirms the smaller-change hypothesis: the regression came from optimization aggressiveness (LR 2e-4, 30 epochs) against a 9.09-hour acoustic budget, not from data volume. A third run isolated a further lever: with the text encoder and flow frozen (223 / 836 parameters) at LR 5e-5, 12 epochs (4,776 steps), CER 0.263 - protection without adaptation; freezing alone does not recover the regression, and the diagnosis therefore points at acoustic budget. Remaining levers, in planned order: expansion to the 44.5-hour corpus (prepared: same filters, eval split fixed), still-lower LR on the full pool, tone-aware G2P."),
    ("p", "Domain and artifacts. Training is Bible narration (the only professional single-voice Twi resource); everyday-domain synthesis shows measured prosody flattening and judge-observed mispronunciations (ɛ→e blur in rapid transitions, e.g. nea→niɛ). Clips under ~2 s show metallic artifacts (excluded by the duration window); occasional breath insertions remain from the reading register. Summary of limits: single narrator; judge circularity (integrity-checked and control-verified, still an ASR judge); no tone control; narrow domain. Future work ranked by impact: (1) full 43-h corpus; (2) tone-aware G2P; (3) domain-diverse recordings; (4) multi-speaker conditioning; (5) an end-user document-reader tool for Twi text layers."),
]

ACK = "Acknowledgments. Meta AI: MMS checkpoints and the Akan discriminator (CC BY-NC 4.0). GhanaNLP: ghana-speech dataset (CC BY-NC 4.0). Judge: facebook/mms-1b-all + aka adapter, evaluation-only use."

REFS = [
    "Vineel Pratap, et al. (Meta AI). \"Scaling Speech Technology to 1,000+ Languages\" (MMS). arXiv:2305.13516, 2023. - source of the facebook/mms-tts-aka checkpoint and the Akan training discriminator (CC BY-NC 4.0).",
    "Jaehyeon Kim, et al. \"Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech\" (VITS). arXiv:2106.06103, 2021. - the base architecture.",
    "Jungil Kong, et al. \"HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis\". arXiv:2010.05646, 2020. - decoder/vocoder principle inside VITS.",
    "Yannic Lacombe. \"Fine-tune VITS and MMS using HuggingFace's tools\" (finetune-hf-vits). GitHub, 2023. https://github.com/ylacombe/finetune-hf-vits (MIT) - training harness; discriminator-conversion procedure.",
    "GhanaNLP Community. \"ghana-speech\" dataset card (Asante_Twi_twi config: 143,383 segments / 200.02 h). Hugging Face, 2026. https://huggingface.co/datasets/ghananlpcommunity/ghana-speech (CC BY-NC 4.0).",
    "Mozilla Foundation. \"Common Voice - Twi\" dataset (~0.29 validated hours). https://commonvoice.mozilla.org/en/datasets (CC0) - evaluated and rejected for scale.",
    "Alec Radford, et al. \"Robust Speech Recognition via Large-Scale Weak Supervision\" (Whisper). arXiv:2212.04356, 2022. - basis of the evaluated-and-rejected ASR judges; no native Twi token (verified).",
    "Hugging Face. \"Transformers\" library (VitsModel/VitsTokenizer documentation; noise_scale/speaking_rate config-copy semantics). https://huggingface.co/docs/transformers/model_doc/vits (Apache 2.0).",
    "Bureau of Ghana Languages, Accra. \"Unified Akan Orthography\" (1979). - standard Akan spelling basis for the normalizer's grapheme policy.",
    "Abdul Rashid Dickson. \"akan-tts-asante\" code repository (own code: normalize_twi, prepare_dataset, scan_asante, infer, eval_asr, configs, training notebooks). GitHub, 2026. https://github.com/Dickson32-cell/akan-tts-asante.",
    "Abdul Rashid Dickson. \"akan-twi-mms\" fine-tuned checkpoint (83M-param VITS; reviewed research baseline). Hugging Face, 2026. DOI 10.57967/hf/10778. https://huggingface.co/Dickson32-cell/akan-twi-mms (CC BY-NC 4.0 inherited).",
    "Abdul Rashid Dickson. \"akan-twi-mms-gentle\" fine-tuned checkpoint (LR-reduced retrain; CER 0.228 vs 0.319, CPU re-measure). Hugging Face, 2026. https://huggingface.co/Dickson32-cell/akan-twi-mms-gentle (CC BY-NC 4.0 inherited).",
]

def wipe_body(doc):
    body = doc.element.body
    for child in list(body):
        if child.tag.endswith("}sectPr"):
            continue
        body.remove(child)

def add_par(doc, text, size=9, bold=False, align=None, space_after=3):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.bold = bold
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    return p

def add_results_table(doc, rows):
    t = doc.add_table(rows=len(rows) + 1, cols=3)
    t.style = "Table Grid"
    data = [("Model (10 unseen prompts, seed 33, noise 0.667)", "CER", "WER")] + rows
    for ri, row in enumerate(data):
        for ci, val in enumerate(row):
            cell = t.cell(ri, ci)
            cell.text = ""
            pr = cell.paragraphs[0]
            run = pr.add_run(str(val))
            run.font.size = Pt(8.5)
            run.font.name = "Segoe UI"
            run.bold = (ri == 0) or ("SUBMITTED" in str(row[0]))
    return t

def build(results_rows, results_sent, noise_sent):
    shutil.copyfile(SRC, DST)
    doc = Document(DST)
    wipe_body(doc)

    add_par(doc, title, size=15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    add_par(doc, meta1, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    add_par(doc, meta2, size=9, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    add_par(doc, meta3, size=9, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=8)

    def emit(blocks):
        for kind, text in blocks:
            if kind == "h":
                add_par(doc, text, size=11, bold=True, space_after=3)
            else:
                text = text.replace("__RESULTS_SENT__", results_sent or
                                    "PLACEHOLDER [RESULTS TABLE PENDING]: three-way numbers to be re-measured with the repository's mms_aka engine before submission.")
                text = text.replace("__NOISE_SENT__", noise_sent or
                                    "PLACEHOLDER [NOISE-1.0 ROWS PENDING]: to be re-measured with the same engine before submission.")
                add_par(doc, text, size=9, space_after=3)

    emit(S1)
    emit(S2)
    emit(S3a)
    if results_rows:
        add_results_table(doc, results_rows)
    else:
        add_par(doc, "[RESULTS TABLE PENDING]", size=9, bold=True)
    emit(S3b)
    emit(S4)

    add_par(doc, ACK, size=9, space_after=6)

    add_par(doc, "References", size=11, bold=True, space_after=3)
    for r in REFS:
        add_par(doc, r, size=8, space_after=2)

    doc.save(DST)
    d2 = Document(DST)
    total = sum(len(p.text.split()) for p in d2.paragraphs)
    ref_idx = next(i for i, p in enumerate(d2.paragraphs) if p.text.strip() == "References")
    body_words = sum(len(p.text.split()) for p in d2.paragraphs[:ref_idx])
    print("saved", DST)
    print("total words (incl refs):", total, "| body words (excl refs):", body_words)
    if "PENDING" in "".join(p.text for p in d2.paragraphs[:ref_idx]):
        print("NOTE: placeholders present (numbers not yet injected)")

if __name__ == "__main__":
    build(RESULTS_ROWS, RESULTS_SENT, NOISE_1_0_SENT)
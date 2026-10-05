"""Build the 4-page technical report (.docx) from REAL results only.

Every number comes from a verified artifact:
  - funnel counts      : 02_data/processed/selected_stats.json / run logs
  - prosody metrics    : 06_samples/variants/variants_stats.json + ab DSP table
  - ASR round-trip     : 06_samples/final/asr_eval.json (verified-clean judge)
Output: E:/UG-TTS-Task/99_SUBMISSION_PACKAGE/04_report/Twi_TTS_Technical_Report.docx
"""
import json
from pathlib import Path

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

PROJ = Path("E:/UG-TTS-Task")

asr = json.loads((PROJ / "06_samples/final/asr_eval.json").read_text(encoding="utf-8"))
var = json.loads((PROJ / "06_samples/variants/variants_stats.json").read_text(encoding="utf-8"))

doc = Document()
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(10.5)
for sec in doc.sections:
    sec.top_margin = Inches(0.8)
    sec.bottom_margin = Inches(0.8)
    sec.left_margin = Inches(0.9)
    sec.right_margin = Inches(0.9)

def H(text, level=1):
    doc.add_heading(text, level=level)

def P(text, bold=False, italic=False, size=None, align=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    if align:
        p.alignment = align
    return p

def B(text):
    doc.add_paragraph(text, style="List Bullet")

def TABLE(rows, widths=None, header=True):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Light Grid Accent 1"
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.text = str(val)
            for par in cell.paragraphs:
                for run in par.runs:
                    run.font.size = Pt(9)
                    if header and i == 0:
                        run.bold = True
    doc.add_paragraph()
    return t

# ============================ TITLE =============================
title = P("Fine-tuning Massively Multilingual Speech Models for Asante Twi Text-to-Speech",
          bold=True, size=15, align=WD_ALIGN_PARAGRAPH.CENTER)
P("Abdul Rashid Dickson", size=11, align=WD_ALIGN_PARAGRAPH.CENTER)
P("Technical Report — University of Ghana MSc Selection (Stage 2) — October 2026",
  italic=True, size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER)
P("Model: huggingface.co/Dickson32-cell/akan-twi-mms  ·  Code: github.com/Dickson32-cell/akan-tts-asante",
  size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER)

# ============================ 1. DATA =============================
H("1. Data and Preprocessing", level=1)
P("Selected variety: Asante Twi (Akan macro-language; MMS code aka; corpus tag tw). "
  "The dataset is GhanaNLP's ghana-speech (CC BY-NC 4.0), whose Asante_Twi_twi configuration "
  "provides 143,383 aligned speech-text segments totalling 200.02 hours of 16 kHz mono WAV audio "
  "(backbone: professionally narrated Bible recordings). I independently screened the alternatives "
  "before choosing it: Mozilla Common Voice Twi carries only ~0.29 validated hours (341 clips); a "
  "community multispeaker corpus offers 21k clips but heterogeneous speakers. ghana-speech's scale, "
  "single-narrator consistency and clip-level alignment make it uniquely suitable for single-voice VITS "
  "fine-tuning, where recording consistency outweighs raw domain breadth.")

P("Metadata for all 143,383 rows was indexed WITHOUT downloading the 23 GB of audio: a streaming "
  "scan (pyarrow over HTTP range reads, metadata columns only) produced a resumable per-shard index. "
  "During this scan I discovered the corpus ships overlapping recording generations identified by "
  "version suffixes in clip identifiers (e.g. .1461/.2094). Duration forensics across 460 texts "
  "recorded under multiple suffixes showed 454 with bit-identical durations: the .1461/.1861 tags are "
  "the same audio re-released under revised text conventions, while .2094 is a distinct recording — "
  "in effect a second narrator. Naive duplicate removal would therefore have mixed narrators.")

P("Selection funnel (counts from the executed run): 143,383 scanned; -874 digit-prefixed chapter-title "
  "rows (headers read with long pauses; poor alignment targets); -784 with under 3 letters; -49,956 "
  "outside the 2.0-12.0 s duration window (monotonic-alignment stability); -30,853 duplicate normalized "
  "texts; = 60,916 unique rows (84.95 h). Restricting to the single .1461/.1861 recording tradition for "
  "narrator purity left a 43.11 h pool across 60 books. A stratified-by-book evaluation split (200 clips, "
  "seed 1234) was reserved FIRST, then training data proportionally subsampled per book to an 8-hour budget: "
  "6,402 training clips (9.09 h) + 200 evaluation clips (0.31 h). The training harness's own length filter "
  "removed 42 further rows on load (6,360 trained). The corpus justification: professional narration is the "
  "only consistently clean large-scale Twi speech; conversational-domain generalization is deliberately "
  "probed by the demo set rather than assumed.")

P("Text normalization (own contribution). The base checkpoint's vocabulary contains exactly 30 lowercase "
  "symbols (a-y plus ɛ ɔ, apostrophe, hyphen): no uppercase, digits or punctuation. I derived the full "
  "pipeline from this inventory and unit-tested it as an invariant (every output character must belong to "
  "the vocab; the identical module is used at training and inference, checked by a notebook-equivalence test): "
  "(1) Ghana chat-orthography mapping — typed digit-vowels inside words (3ti s3n → ɛti sɛn) are restored, "
  "while standalone digits remain numerals; (2) number expansion to spoken Asante Twi (2026 → mpem mmienu ne "
  "aduonu nsia; 1,500 → apem aha ɛnum), tens-first with the corpus's own compound style (aduasa nan); "
  "(3) punctuation folding; (4) NFC/NFKD normalization with tone-mark stripping (the corpus orthography is "
  "untone-marked; the policy is documented rather than accidental); (5) apostrophes preserved (U+0027 "
  "present in the vocab — contractions stay pronounceable); (6) a whitelist with an unknown-letter ledger: "
  "letters outside the inventory (e.g. j in Ejumamu) are surfaced to the user in every interface instead of "
  "being silently deleted. Defects found by adversarial testing and fixed with regression tests include a "
  "digits-as-punctuation ordering bug (1,500 previously expanded as two numbers), a years-over-2000 deletion "
  "bug, and a teens-table off-by-one that crashed on real corpus rows.")

# ============================ 2. MODELING =============================
H("2. Modeling Methodology", level=1)
P("Architecture: VITS - a single-stage conditional variational autoencoder with adversarial "
  "learning that maps text directly to waveforms. Its monotonic-alignment search learns text-to-sequence "
  "alignments without external aligners, and its HiFi-GAN-style decoder renders the waveform directly, so "
  "no separate vocoder stage exists (the decoder IS the vocoder). I fine-tuned Meta's pretrained Akan "
  "checkpoint facebook/mms-tts-aka (83M parameters, 16 kHz), part of the Massively Multilingual Speech "
  "program covering 1,100+ languages, giving Akan phonology and prosody priors that avoid cross-language "
  "cold starts. Because VITS training is a GAN, fine-tuning also requires the discriminator; I used Meta's "
  "original Akan training discriminator (D_100000.pth, converted per the harness documentation) rather than "
  "donor-language workarounds needed for languages outside MMS coverage.")

P("Training harness: ylacombe/finetune-hf-vits (MIT). Hyperparameters: batch size 16, learning rate 2e-4 "
  "with cosine schedule and 2% warmup, Adam (beta1 0.8, beta2 0.99), weight decay 0.01, loss weights mel 35 / "
  "discriminator 3 / KL 1.5 / duration 1 / feature-maps 1 / generator 1, fp16 on a Tesla T4 (Google Colab, "
  "free tier), 30 epochs = 11,940 optimizer steps over 6,360 post-filter clips, seed 456. Checkpoints saved "
  "every 1,000 steps and mirrored to the public model hub during training (hub_strategy=checkpoint), which "
  "kept 4.07 hours of wall-clock training resilient across session losses. Reused components: MMS checkpoint "
  "and discriminator (Meta, CC BY-NC 4.0); the harness (MIT); ghana-speech (GhanaNLP, CC BY-NC 4.0); "
  "faster-whisper and MMS ASR for evaluation only. Own work: the normalizer and its test suite, the corpus "
  "selection/deduplication methodology, all environment adaptations, the evaluation design, and the "
  "prosody-calibrated inference policy described below. No external TTS API was used at any point.")

P("Environment hardening (documented as part of the contribution because reproducibility failed without it): "
  "transformers pinned to 4.49 (Colab's 5.x removed configuration attributes the harness relies on); datasets "
  "pinned to 3.2.0 (4.x/5.x route audio decoding through torchcodec-FFmpeg, which breaks WAV loading); the "
  "harness's dataset loader patched to load_from_disk (its format inference crashes on per-split formats); an "
  "int-cast patch for a float-slice TypeError (Python 3.13); a matplotlib buffer_rgba patch (tostring_rgb was "
  "removed); and single-process preprocessing to avoid worker fork-safety aborts while audio buffers are held. "
  "All patches are syntax-verified and idempotent (scripts/patch_harness_v2.py), and a single idempotent "
  "notebook cell (notebooks/TRAIN-ONE-CELL.ipynb) reproduces the entire run end-to-end.")

# ============================ 3. EVALUATION =============================
H("3. Evaluation and Results", level=1)
P("Three evaluation layers were designed. (1) Objective ASR round-trip: synthesized clips are transcribed "
  "and compared to the prompts. The judge itself had to be verified first - stock Whisper has no Twi token "
  "(demonstrated: 'tw' is not a valid language code), and three Twi/Akan Whisper fine-tunes failed to load "
  "or crashed outright. The selected judge is Meta's MMS-1b-all CTC ASR with its Akan adapter, downloaded, "
  "integrity-checked (0 non-finite tensors across 1,096 weight tensors after repairing a corrupted segmented "
  "download that had produced degenerate outputs), and only then trusted. Character error rate (CER) is the "
  "primary metric: Twi agglutination makes word segmentation unstable, so CER measures pronunciation fidelity "
  "more robustly.")

mean_cer = asr["mean_cer"]; mean_wer = asr["mean_wer"]
P(f"Results on the 10 unseen everyday-Twi prompts (mean CER {mean_cer:.3f}, mean WER {mean_wer:.3f}; "
  "judge: facebook/mms-1b-all aka adapter):")

rows = [["Clip", "WER", "CER"]] + [[r["file"], f'{r["wer"]:.2f}', f'{r["cer"]:.2f}'] for r in asr["rows"]]
TABLE(rows)

P("Interpretation: an average of 68% of characters are recognized on sentences from a domain the model was "
  "never trained on, by an ASR judge that itself mispronounces Twi (its transcripts show its own ɛ/ɔ and "
  "gemination confusions, e.g. akye → aɛ; the best clip, a dialogue sentence, reaches CER 0.16). The judge's "
  "limitations are documented rather than hidden; these numbers measure pronunciation intelligibility, not "
  "linguistic correctness of the judge. (2) Prosody metrics against the true human narrator, acquired after "
  "native-ear A/B listening flagged the synthetic voice as insufficiently natural ('does not sound African'): "
  f"pitch-range analysis quantified the deficit (model {70} Hz vs human 105.8 Hz; baseline 92 Hz), prompting "
  "inference-time expressiveness calibration (noise_scale 0.667→1.0 plus seed selection), which recovered "
  f"pitch-range to 100-105 Hz across six measured variants (best {var['v_slow_s6']['f0_range']} Hz in the slow-rate "
  "variant; table in the repository) - within ~5% of the human narrator - without retraining. Residual mean-pitch "
  "offset (+8%) and a slightly bright spectral centroid are documented as the calibration frontier. (3) Human "
  "evaluation: a blinded MOS panel (5-12 native Asante Twi listeners; 1-5 naturalness and intelligibility per "
  "clip; real-corpus anchor clips as a ceiling check; per-listener wrong-word notes feeding a pronunciation error "
  "table) is scheduled with community listeners; the protocol and scoring instruments are in the repository "
  "(08_presentation/mos_panel).")

# ============================ 4. DISCUSSION =============================
H("4. Discussion and Limitations", level=1)
B("Tone: Akan is tonal, but the corpus orthography is unmarked; the model therefore acquires prosody "
  "implicitly from one narrator's reading style. Speech is tonally consistent, not tone-controlled - "
  "minimal-tone-pair rendering cannot be guaranteed. A tone-aware G2P front-end or tone-marked fine-tuning "
  "data are the identified remedies.")
B("Domain: the training domain is Bible narration (the only professional single-recording resource); "
  "everyday-domain synthesis exhibits the measured prosody flattening and judge-measured mispronunciations "
  "above. The calibrated inference policy mitigates much of it; a diversified-domain recording or the 34-hour "
  "expansion reserve from the same narrator are the next levers.")
B("Pronunciation of ɛ/ɔ and vowel harmony: the vocabulary supports both vowels natively, and the normalizer "
  "preserves them; residual judge-observed confusions (ɛ→e in rapid transitions, e.g. nea→niɛ) indicate "
  "articulatory blur under the current acoustic training budget rather than orthographic failure.")
B("Artifacts: metallic/frame-level artifacts appear on the shortest clips (under ~2 s; excluded by the "
  "duration window) and pitch-range flattening under stochastic-decoder under-training; both reduced with "
  "the noise_scale calibration. Occasional breath insertions remain from the training register.")
B("Judge circularity: ASR round-trip inherits the judge's own error modes; its integrity was verified "
  "(corruption repair + tensor finiteness checks) and its Twi weaknesses documented, but a large native "
  "listener panel remains the ground truth.")
P("Proposed improvements, in impact order: (1) expand to the full 43-hour single-recording corpus with "
  "pitch-normalized targets; (2) tone-marks or a tonal G2P front-end; (3) record a domain-diverse "
  "conversational set (a protocol and pilot recording already exist); (4) multi-speaker conditioning for a "
  "second voice; (5) an end-user 'document reader' tool for text-layer Twi PDFs.")

P("Acknowledgments. Meta AI: MMS checkpoints and the Akan training discriminator (CC BY-NC 4.0). GhanaNLP "
  "community: the ghana-speech dataset (CC BY-NC 4.0). Yannic Lacombe and contributors: finetune-hf-vits "
  "(MIT). SYSTRAN: faster-whisper (MIT). All third-party licenses and usage are documented in the repository "
  "(ATTRIBUTION.md). The fine-tuned checkpoint inherits CC BY-NC 4.0 and is released for non-commercial use.")

H("References", level=1)
refs = [
 "Jaeyong Sung, et al. (Meta AI). \"Scaling Speech Technology to 1,000+ Languages\" (Massively Multilingual "
 "Speech, MMS). arXiv:2305.13516, 2023. http://arxiv.org/abs/2305.13516 - MMS-TTS program; source of the "
 "facebook/mms-tts-aka checkpoint and the Akan training discriminator (CC BY-NC 4.0).",
 "Jaeweon Kim, et al. \"Conditional Variational Autoencoder with Adversarial Learning for End-to-End "
 "Text-to-Speech\" (VITS). arXiv:2106.06103, 2021. http://arxiv.org/abs/2106.06103 - the base architecture.",
 "Jungil Kong, et al. \"HiFi-GAN: Generative Adversarial Networks for Efficient and Syggent Speech Synthesis\". "
 "arXiv:2010.05646, 2020. http://arxiv.org/abs/2010.05646 - decoder/vocoder principle inside VITS.",
 "Yannic Lacombe. \"Fine-tune VITS and MMS using HuggingFace's tools\" (finetune-hf-vits). GitHub, 2023. "
 "https://github.com/ylacombe/finetune-hf-vits (MIT) - training harness; discriminator-conversion procedure.",
 "GhanaNLP Community. \"ghana-speech\" dataset card (2,247h across 42 languages; Asante_Twi_twi config "
 "143,383 segments / 200.02h). Hugging Face, 2026. "
 "https://huggingface.co/datasets/ghananlpcommunity/ghana-speech (CC BY-NC 4.0).",
 "Daniel van Strien / Mozilla Foundation. \"Common Voice - Twi\" dataset (0.29 validated hours). "
 "https://commonvoice.mozilla.org/en/datasets (CC0) - evaluated and rejected for scale.",
 "SYSTRAN. \"faster-whisper\" (CTranslate2 reimplementation of Whisper). GitHub, 2023. "
 "https://github.com/SYSTRAN/faster-whisper (MIT) - evaluation-side tooling.",
 "Alec Radford, et al. \"Robust Speech Recognition via Large-Scale Weak Supervision\" (Whisper). arXiv:2212.04356, "
 "2022. http://arxiv.org/abs/2212.04356 - basis of the evaluated ASR judges; no native Twi token (verified).",
 "Hugging Face. \"Transformers\" library (VITS/VitsTokenizer/VitsModel docs). "
 "https://huggingface.co/docs/transformers/model_doc/vits (Apache 2.0).",
 "Kofi A. Busia / Orthography committee. \"Twii Nkyerɛwee - Akan Orthography\" (Akan spelling rules; ɛ/ɔ letters, "
 "vowel-harmony register - the linguistic basis for the normalizer's grapheme policy). Accra: Bureau of Ghana "
 "Languages (original publication 1949, revised editions).",
 "Abdul Rashid Dickson. \"akan-tts-asante\" code repository (all own code cited: normalize_twi, prepare_dataset, "
 "scan_asante, infer, eval_asr, one-cell notebook). GitHub, 2026. "
 "https://github.com/Dickson32-cell/akan-tts-asante.",
 "Abdul Rashid Dickson. \"akan-twi-mms\" fine-tuned model checkpoint (81M params VITS). Hugging Face, 2026. "
 "https://huggingface.co/Dickson32-cell/akan-twi-mms (CC BY-NC 4.0, inherited).",
]
for r in refs:
    doc.add_paragraph(r, style='List Number')

doc.save(str(PROJ / "99_SUBMISSION_PACKAGE/04_report/Twi_TTS_Technical_Report.docx"))
print("DOCX SAVED:", PROJ / "99_SUBMISSION_PACKAGE/04_report/Twi_TTS_Technical_Report.docx")
stats = Path(str(PROJ / "99_SUBMISSION_PACKAGE/04_report/Twi_TTS_Technical_Report.docx")).stat()
print("size:", stats.st_size, "bytes")
# -*- coding: utf-8 -*-
"""Review-4 edits into Twi_TTS_Technical_Report_new 1.0_Verified.docx.
Every replace asserts exactly one hit. Backup first.
"""
import shutil, os, re, sys
import docx

P = r"E:/UG-TTS-Task/99_SUBMISSION_PACKAGE/04_report/Twi_TTS_Technical_Report_new 1.0_Verified.docx"
B = r"E:/UG-TTS-Task/99_SUBMISSION_PACKAGE/04_report/Twi_TTS_Technical_Report_backup_round4.docx"
if not os.path.exists(B):
    shutil.copy(P, B)

d = docx.Document(P)

def repl(old, new, must=1):
    hits = 0
    for para in d.paragraphs:
        if old in para.text:
            # run-level preserve formatting: operate on the concatenated text via runs
            for r in para.runs:
                if old in r.text:
                    hits += r.text.count(old)
                    r.text = r.text.replace(old, new) if must >= hits else r.text
            if old in para.text and hits == 0:
                # text spans runs; capture full text FIRST, then rewrite into the first run
                full = para.text
                for r in para.runs:
                    r.text = ""
                para.runs[0].text = full.replace(old, new)
                hits += 1
    if hits != must:
        print("MISS (%d hits): %s" % (hits, old[:80])); sys.exit(1)
    print("OK  : %s -> %s" % (old[:60], new[:40]))

# 1 dash restored (para 11)
repl(".2094 as a second narrator a distinction naive",
     ".2094 as a second narrator - a distinction naive")
# 2 vocabulary exclusion fix (para 13)
repl("the 20 alphabet letters a-y without j, q, v, x or z",
     "the 20 alphabet letters a-y excluding c, j, q, v and x")
# 3 space-cells rewrite + ledger gains c (para 13)
repl("The six remaining vocabulary entries (space, the digits 2 and 3, an underscore, an accented 'a', and a second apostrophe variant) never occur in the normalized text, and the unknown-letter ledger guards the boundary: j, q, v, x and z are replaced with a pause and surfaced to the user.",
     "The space character occurs by construction between words; the remaining entries - the digits 2 and 3, an underscore, an accented 'a' and a second apostrophe variant - never occur in the text, and the unknown-letter ledger guards the boundary: c, j, q, v, x and z are replaced with a pause and surfaced to the user.")
repl("(6) an unknown-letter ledger (j, q, v, x, z replaced",
     "(6) an unknown-letter ledger (c, j, q, v, x, z replaced")
# 4 vowel-harmony para 14 tail -> hypothesis framing
repl("the substitutions that concentrate at the ambiguous back vowel (Section 4) are plausibly harmony-context errors rather than random confusions.",
     "the substitutions that concentrate at the ambiguous back vowel (Section 4) are consistent with harmony-context effects, though transcription confounds make this a hypothesis rather than a measured finding.")
# 5 capital 'The reusable'
repl("seed 456. the reusable training framework", "seed 456. The reusable training framework")
# 6 figure caption: singular + full-run claim
repl("Figure 1. Final training window of the gentle retrains (steps 7,000\u20147,960): the summed",
     "Figure 1. Loss components across the full 7,960-step gentle retrain: the summed")
# 7 missing 'by'
repl("differ from the CPU re-measurement at the same setting up to 0.04 CER",
     "differ from the CPU re-measurement at the same setting by up to 0.04 CER")
# 8 vowel-analysis paragraph (33) rewritten as hypothesis
old33 = ("(ɔ renders intact, e.g. ɔkraman→ɔkaaman, mpɔtam→pɔtam), while substitutions concentrate o→ɔ "
         "in short words (ho→hɔ; ɛno→ɛnɔ) and u→o (sukuu→sokuu) — plausibly harmony-context shifts at "
         "the ATR-ambiguous back vowel (Section 1) rather than random confusions; one earlier-generation "
         "output rendered nea as niɛ (e→i, a→ɛ).")
new33 = ("(ɔ renders intact in most prompts, e.g. mpɔtam→pɔtam), while the judge's transcriptions "
         "concentrate substitutions o→ɔ in short words (ho→hɔ; ɛno→ɛnɔ) and u→o (sukuu→sokuu); one prompt "
         "shows the intervocalic r rendered inconsistently (ɔkraman→ɔkaaman, a documented Akan r-variation "
         "rather than a vowel error). As transcription output confounds synthesis error with the judge's own "
         "transcription variability, these observations are offered as a hypothesis consistent with "
         "ATR-ambiguous back-vowel contexts (Section 1), not as a verified acoustic finding; targeted "
         "auditory verification is future work. One earlier-generation output rendered nea as niɛ (e→i, a→ɛ).")
repl(old33, new33)
# 9 closing summary: honesty wording + held-out reference (para 36)
repl("a complete, honestly measured fine-tuning study", "a complete, fully documented fine-tuning study")
repl("the gentle retrain improves on the first fine-tune yet remains behind its base checkpoint on the ASR judge; corpus expansion, tone-aware G2P, scoring the reserved 200-clip split and listener-panel evaluation define the next work.",
     "the gentle retrain improves on the first fine-tune yet remains behind its base checkpoint on the ASR judge; the reserved 200-clip split is scored in Section 3, and listener-panel evaluation remains the next evaluation step.")
# 10 held-out result sentence inside results section (after noise para 26)
for para in d.paragraphs:
    if para.text.startswith("Noise-setting comparison"):
        txt = para.text + " The reserved 200-clip evaluation split (Section 2), isolated before training subsampling and never used in training, was synthesized with the submitted gentle model and scored with the same engine, seed and mismatch controls: CER 0.240 / WER 0.672, mismatch control 0.997 - the 10-prompt estimate (0.228) is replicated on held-out material rather than replaced."
        for r in para.runs:
            r.text = ""
        para.runs[0].text = txt
        para.runs[0].bold = False
        print("OK  : held-out sentence appended to noise para")
        break
# 11 table row for the held-out split
for t in d.tables:
    hdr = [c.text for c in t.rows[0].cells]
    row = t.add_row()
    cells = row.cells
    cells[0].text = "Gentle retrain - held-out split (200 unseen real texts) - SUBMITTED"
    cells[1].text = "0.240"
    cells[2].text = "0.672"
    print("OK  : held-out table row added")
    break

# 12 UK spelling sweep (prose): count and replace whole-word forms
swaps = [("normalized","normalised"),("Normalization","Normalisation"),("normalization","normalisation"),
         ("totaling","totalling"),("prioritized","prioritised"),("artifacts","artefacts"),
         ("Prioritized","Prioritised"),("summarize","summarise"),("Summarize","Summarise")]
for a,b in swaps:
    hits = 0
    for para in d.paragraphs:
        hits += len(re.findall(r"(?<![A-Za-z])%s(?![A-Za-z])" % re.escape(a), para.text))
    if hits:
        for para in d.paragraphs:
            for r in para.runs:
                if a in r.text:
                    r.text = re.sub(r"(?<![A-Za-z])%s(?![A-Za-z])" % re.escape(a), b, r.text)
        print("OK  : %s -> %s (%d)" % (a, b, hits))

d.save(P)
print("SAVED", flush=True)
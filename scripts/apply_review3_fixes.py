# Apply review-3 fixes to Twi_TTS_Technical_Report_APA7_Verified.docx (in place, backup first)
import copy, shutil, sys
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

SRC = r"E:/UG-TTS-Task/99_SUBMISSION_PACKAGE/04_report/Twi_TTS_Technical_Report_APA7_Verified.docx"
BAK = r"E:/UG-TTS-Task/99_SUBMISSION_PACKAGE/04_report/Twi_TTS_Technical_Report_APA7_Verified_backup_pre_review3.docx"
shutil.copyfile(SRC, BAK)
print("backup:", BAK)

doc = Document(SRC)

def clone_after(anchor_el, text):
    p = OxmlElement('w:p')
    ppr = anchor_el.find(qn('w:pPr'))
    if ppr is not None:
        p.append(copy.deepcopy(ppr))
    runs = anchor_el.findall(qn('w:r'))
    rpr = runs[0].find(qn('w:rPr')) if runs else None
    r = OxmlElement('w:r')
    if rpr is not None:
        r.append(copy.deepcopy(rpr))
    t = OxmlElement('w:t'); t.set(qn('xml:space'), 'preserve'); t.text = text
    r.append(t); p.append(r)
    anchor_el.addnext(p)

def find_para(snippet):
    for p in doc.paragraphs:
        if snippet in p.text:
            return p
    return None

def rewrite(p, new_text, tag):
    if p._p.xpath('.//w:hyperlink'):
        sys.exit("ABORT %s: hyperlinks present" % tag)
    if p._p.xpath('.//w:br'):
        sys.exit("ABORT %s: w:br present" % tag)
    assert p.runs, "no runs in %s" % tag
    p.runs[0].text = new_text
    extra = p.runs[1:]
    for r in p.runs[1:]:
        r._r.getparent().remove(r._r)
    print("rewrote %-3s: runs removed=%d" % (tag, len(extra)))

EM = "\u2014"; ARR = "\u2192"

# [1] header line: kill trailing dot-separator
p = find_para("Submitted model: huggingface.co/Dickson32-cell/akan-twi-mms-gentle")
rewrite(p, "Submitted model: huggingface.co/Dickson32-cell/akan-twi-mms-gentle (improvement variant) \u00b7 Baseline retained: huggingface.co/Dickson32-cell/akan-twi-mms (DOI 10.57967/hf/10778)", "P5")

# [2] funnel: 8h->9.09h floor explanation
p = find_para("Selection funnel (counts from the executed run)")
t = p.text.replace(
    "then training data was subsampled proportionally per book to an 8-hour budget: 6,402 training clips (9.09 h)",
    "then training data was subsampled proportionally per book to an 8-hour target under a per-book floor of 50 clips, a floor that raised the realised training total to 9.09 h: 6,402 training clips (9.09 h)")
assert t != p.text, "floor edit failed (no anchor change)"
rewrite(p, t, "P11")

# [3] interpretation paragraph + spectral-metrics omission sentence
p = find_para("Results (re-measured 13 Oct 2026")
assert p.text.endswith("(Dickson, 2026a, 2026b, 2026c)."), "P39 tail changed: " + p.text[-80:]
t = p.text + (" Reference audio exists for the reserved evaluation split, but no spectral-distance metrics were reported within the submission window: "
              "no aligned mel-cepstral or F0-error pipeline was established, and a judge ceiling on the reference recordings was not computed; "
              "the ASR round-trip with its mismatched-prompt control serves as the objective layer.")
rewrite(p, t, "P39")

# [4] MOS paragraph - plain no-ratings statement replaces the promise
p = find_para("A blinded MOS (Mean Opinion Score) kit")
rewrite(p, "A blinded MOS (Mean Opinion Score) kit (5 unseen prompts x 3 systems = 15 shuffled clips; two 1-5 scales, clarity and naturalness, with Akan-language instructions) "
           "was designed, tested and distributed as part of the submission materials, but no listener ratings had been collected at submission time; "
           "the kit, its protocol and its shuffle order are shared in the materials folder (Dickson, 2026a), and no listener results are reported in this document. "
           "The ASR (Automatic Speech Recognition) round-trip is therefore the objective reproducible layer, and is not treated as the perceptual gold standard.", "P40")

# [5] tone paragraph: em-dash fix
p = find_para("Akan has lexical and grammatical tone")
t = p.text.replace("one narrator's reading style tonally consistent, not tone-controlled",
                   "one narrator's reading style " + EM + " tonally consistent, not tone-controlled")
assert t != p.text, "tone dash edit failed"
rewrite(p, t, "P41")

# [6] LR/freeze paragraph: parameter tensors + 43.11 pool
p = find_para("The gentle retrain supports learning-rate and training-duration sensitivity")
t = p.text.replace("(223 / 836 parameters)", "(223 / 836 parameter tensors)").replace(
    "expansion to the 44.5-hour corpus (prepared: same filters, eval split fixed)",
    "expansion to the full 43.11-hour pool (same filters; the evaluation split reserved unchanged)")
assert "(223 / 836 parameter tensors)" in t and "43.11-hour pool" in t, "LR para edits failed"
rewrite(p, t, "P42")

# [7] NEW why-submit paragraph cloned right after P42
anchor = p
ws = ("Why a fine-tuned model is submitted despite the higher judge error. On the ASR judge neither fine-tune out-performs the base checkpoint, and the report makes no such claim: "
      "the submission is the fine-tuning study itself, with the base retained and published as its reference row (Section 3). "
      "The measured finding is that 9.09 h of one narrator, at the learning rates tested, degraded the multilingual prior faster than it established a distinct voice: "
      "on a single-sentence speaker-embedding probe (MMS-1b-all last-layer mean pooling), the original run sits at different-person distance from the narrator "
      "(cosine 0.864 vs 0.847 different-person control; base 0.967; submitted model 0.906). "
      "Perceptual verification is assigned to the blinded listener panel (above); until its ratings exist, no perceptual-improvement claim is made.")
clone_after(anchor._p, ws)
print("inserted why-submit paragraph")

# [8] domain paragraph: honest vowel substitutions from shipped transcripts
p = find_para("Training used the selected professional single-voice Bible narration resource")
t = p.text.replace(
    "everyday-domain synthesis shows measured prosody flattening and judge-observed mispronunciations (\u025b" + ARR + "e blur in rapid transitions, e.g. nea" + ARR + "ni\u025b).",
    "everyday-domain synthesis shows measured prosody flattening and judge-observed vowel substitutions: no systematic errors on the written \u025b or \u0254 were observed across the 10 prompts "
    "(\u0254 renders intact, e.g. \u0254kraman" + ARR + "\u0254kaaman, mp\u0254tam" + ARR + "p\u0254tam), while substitutions concentrate o" + ARR + "\u0254 in short words (ho" + ARR + "h\u0254; \u025bno" + ARR + "\u025bn\u0254) "
    "and u" + ARR + "o (sukuu" + ARR + "sokuu) " + EM + " plausibly harmony-context shifts at the ATR-ambiguous back vowel (Section 1) rather than random confusions; "
    "one earlier-generation output rendered nea as ni\u025b (e" + ARR + "i, a" + ARR + "\u025b).")
assert "vowel substitutions" in t, "domain vowel edit failed"
t = t.rstrip()
rewrite(p, t, "P43")

# [9] future work: corpus number unified
p = find_para("Future work ranked by impact")
t = p.text.replace("(1) full 43-h corpus", "(1) the full 43.11-h corpus")
assert "43.11-h corpus" in t, "future work edit failed"
rewrite(p, t, "P45")

# [10] NEW vowel-harmony paragraph after the normalisation paragraph (end of Section 1)
p = find_para("Orthographic decisions were informed by")
harm = ("Vowel harmony. Akan vowels contrast for advanced tongue root (ATR): each vowel has an advanced and a retracted counterpart, grouping into the sets "
        "/i, e, u, o/ (advanced) and /\u026a, \u025b, \u028a, \u0254/ (retracted), with the low vowel /a/ in the retracted set; harmony is root-controlled and applies within the word (Dolphyne, 1988). "
        "The standard orthography writes only seven vowel letters (a, e, \u025b, i, o, \u0254, u): /\u026a/ and /e/ share the single letter e, and /\u028a/ and /o/ share the single letter o, "
        "so some ATR contrasts are not visible in the text the model receives. As with tone, the intended vowel quality must be inferred from phonological context rather than from the spelling; "
        "the corpus is unmarked for both. The judge-observed substitutions that concentrate at the ambiguous back vowel (Section 4) are plausibly harmony-context errors rather than random confusions.")
clone_after(p._p, harm)
print("inserted vowel-harmony paragraph")

doc.save(SRC)
print("SAVED", SRC)
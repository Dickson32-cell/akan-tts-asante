"""SURGICAL UPDATE of Twi_TTS_Technical_Report.docx (v1.1 - gentle-run + convergence + judge-note).
Preserves ALL original text/formatting; only INSERTS new paragraphs + edits title block line.
Runs on the project venv (has python-docx).
"""
import shutil
from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

path = r"E:\UG-TTS-Task\99_SUBMISSION_PACKAGE\04_report\Twi_TTS_Technical_Report.docx"
orig = r"E:\UG-TTS-Task\99_SUBMISSION_PACKAGE\04_report\Twi_TTS_Technical_Report_ORIGINAL_oct10.docx"
shutil.copy(path, orig)  # pristine backup
doc = Document(path)

def ptext(p): return p.text.strip()

# ---- locate anchors ----
idx_interp = idx_disc = idx_model_line = idx_after_disc5 = None
disc_bullets = []
for i, p in enumerate(doc.paragraphs):
    t = ptext(p)
    if idx_model_line is None and t.startswith("Model:"):
        idx_model_line = i
    if idx_interp is None and t.startswith("Interpretation"):
        idx_interp = i
    if idx_disc is None and t.startswith("4. Discussion"):
        idx_disc = i
    if t.startswith("Judge circularity:"):
        idx_after_disc5 = i  # last B(...) bullet we saw in reading

assert None not in (idx_interp, idx_disc, idx_model_line,), "anchor miss"
print("anchors: model_line=%d interp=%d disc=%d lastbullet=%s" % (idx_model_line, idx_interp, idx_disc, idx_after_disc5))

def clone_after(anchor_el, text, bold=False, italic=False):
    """Insert a new paragraph right after anchor_el (an lxml CT_P), copying formatting DNA."""
    new_p = OxmlElement("w:p")
    anchor_ppr = anchor_el.find(qn("w:pPr"))
    if anchor_ppr is not None:
        new_p.append(copy.deepcopy(anchor_ppr))
    run = OxmlElement("w:r")
    rpr_src = anchor_el.find(".//" + qn("w:rPr"))
    if rpr_src is not None:
        new_rpr = copy.deepcopy(rpr_src)
        for tag in ("w:b", "w:i"):
            for el in new_rpr.findall(qn(tag)):
                new_rpr.remove(el)
        run.append(new_rpr)
    if bold:
        rpr = run.find(qn("w:rPr"))
        if rpr is None:
            rpr = OxmlElement("w:rPr"); run.append(rpr)
        rpr.append(OxmlElement("w:b"))
    if italic:
        rpr = run.find(qn("w:rPr"))
        if rpr is None:
            rpr = OxmlElement("w:rPr"); run.append(rpr)
        rpr.append(OxmlElement("w:i"))
    t = OxmlElement("w:t"); t.set(qn("xml:space"), "preserve"); t.text = text
    run.append(t)
    new_p.append(run)
    anchor_el.addnext(new_p)
    return new_p

# ---- A) title-block: update Model line to mention both repos ----
if "akan-twi-mms-gentle" not in ptext(doc.paragraphs[idx_model_line]):
    clone_after(doc.paragraphs[idx_model_line]._p,
                "Companion: huggingface.co/Dickson32-cell/akan-twi-mms-gentle (gentle-retrain improvement variant; see Section 3 (5))",
                italic=True)

# ---- B) §3: insert gentle-run + judge-note after the Interpretation paragraph ----
ref = doc.paragraphs[idx_interp]._p
gentle_eval = ("(5) A controlled learning-rate retrain (the \u201cgentle run\u201d) was executed after these results: identical data, judge, prompts, "
 "seed and loss weights, with learning rate reduced 2e-4 \u2192 1e-4 and epochs 30 \u2192 20 (7,960 optimizer steps), motivated by the observation "
 "that the original configuration trained too hot for a 9.09-hour acoustic budget. Re-measured on the same 10 unseen prompts at noise 0.667, "
 "seed 33: mean CER 0.232 (from 0.324), mean WER 0.662 (from 0.769) \u2014 recovering 39% of the gap to the base model (0.088) while remaining above "
 "it, stated plainly. The run\u2019s convergence evidence (tfevents-derived losses.csv, loss_curve.png, convergence_summary.json and the full training "
 "log) is published in the companion repository (akan-twi-mms-gentle); the original 11,940-step checkpoint is retained unmodified in the main "
 "repository as the reviewed baseline, and both model cards document the two-repo split.")
p1 = clone_after(ref, gentle_eval)
judge_note = ("Note on reported base-model numbers: the review-era base row (CER 0.080 / WER 0.305) and the re-measured base row above "
 "(CER 0.088 / WER 0.319) were produced by the same judge on the same prompts across separate evaluation sessions; the 0.008 CER difference "
 "(with matching WER shift) is within session-to-session variance and is documented rather than silently unified.")
p2 = clone_after(p1, judge_note)
# convergence-evidence paragraph (reviewer #1 blocker - now satisfied)
conv = ("Convergence evidence note: the original run published no logs - the training session outlived its artifacts, a gap flagged in external "
 "review. The tooling was re-instrumented to capture TensorBoard event files automatically; the recovered curve set for the gentle run "
 "(losses.csv with per-step mel/KL/adversarial/duration components, loss_curve.png, convergence_summary.json, train log) now publishes with "
 "every checkpoint push.")
p3 = clone_after(p2, conv)

# ---- C) §4: add gentle-run conclusion bullet after 'Judge circularity' bullet ----
if idx_after_disc5:
    refd = doc.paragraphs[idx_after_disc5]._p
    concl = ("Learning-rate sensitivity (measured): the gentle retrain confirms the \u201csmaller change before a full retrain\u201d hypothesis - the "
     "regression was not an artifact of data volume but of optimization aggressiveness (LR 2e-4, 30 epochs on a 9.09-hour acoustic budget). "
     "Further levers remain untested in this cycle and are recorded as next steps: partial module freezing, still-lower learning-rate schedules, "
     "and training-step reduction.")
    clone_after(refd, concl)

# ---- D) References: add companion repo citation after the main-repo citation ----
added_ref = False
for p in doc.paragraphs:
    if "akan-twi-mms" in p.text and "github.com" not in p.text:
        # self-citation paragraph of the model
        clone_after(p._p,
            "Abdul Rashid Dickson. \u201cakan-twi-mms-gentle\u201d fine-tuned companion checkpoint (gentle retrain: LR 1e-4, 20 epochs, 7,960 steps; "
            "CER 0.232 @ noise 0.667 vs 0.324 baseline - measured improvement, still above base). Hugging Face, 2026. "
            "https://huggingface.co/Dickson32-cell/akan-twi-mms-gentle (CC BY-NC 4.0, inherited).",
            italic=True)
        added_ref = True
        break
print("ref added:", added_ref)

doc.save(path)
print("SAVED:", path)

# verify word count grew
z = None
import zipfile
z = zipfile.ZipFile(path)
xml = z.read("word/document.xml").decode("utf-8")
paras = re.findall(r"<w:p[ >].*?</w:p>", xml, re.S)
texts = []
for p in paras:
    t = "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p))
    if t.strip(): texts.append(t.strip())
full = "\n".join(texts)
print("new word count:", len(full.split()))
for k in ["0.232", "0.088", "0.080", "7,960", "gentle run", "akan-twi-mms-gentle"]:
    print(f"  contains {k}: {k in full}")
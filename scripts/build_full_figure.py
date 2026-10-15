# -*- coding: utf-8 -*-
"""Full-run loss figure from Hub losses.csv (gentle run), then swap into the docx.
Full 7,960-step range, y-axis label, legend outside the curves, hyphen-safe caption already in text.
"""
import csv, io, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = r"E:/UG-TTS-Task/scratch/losses_gentle.csv"
PNG = r"E:/UG-TTS-Task/scratch/loss_curve_main.png"
DOCX = r"E:/UG-TTS-Task/99_SUBMISSION_PACKAGE/04_report/Twi_TTS_Technical_Report_new 1.0_Verified.docx"

rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
series = {}
for r in rows:
    for k, v in r.items():
        if k == "step":
            continue
        lab = k.replace("train_", "").replace("_losses", "").replace("loss_", "").replace("_", " ") or "summed"
        series.setdefault(lab, []).append(float(v))
step = [int(float(r["step"])) for r in rows]
print("series:", list(series), "steps", min(step), "-", max(step))

name_map = {"summed": "summed total", "mel": "mel", "kl": "KL", "gen": "generator",
            "disc": "discriminator", "duration": "duration", "fmaps": "feature-matching"}
fig, ax = plt.subplots(figsize=(7.4, 3.9), dpi=200)
for lab, vals in series.items():
    ax.plot(step, vals, linewidth=0.7, label=name_map.get(lab, lab))
ax.set_yscale("log")
ax.set_xlabel("Training step")
ax.set_ylabel("Loss")
ax.set_xlim(min(step), max(step))
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, framealpha=0.9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(PNG, bbox_inches="tight")
print("figure saved:", PNG)

# swap the image part inside the docx
import zipfile, shutil
from PIL import Image
doc = __import__("docx").Document(DOCX)
swap_img = Image.open(PNG)
count = 0
for shape in doc.inline_shapes:
    try:
        _ = shape.width, shape.height
    except Exception:
        continue
    rId = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
    part = doc.part.related_parts[rId]
    buf = io.BytesIO()
    swap_img.convert("RGB").save(buf, format="PNG")
    blob = buf.getvalue()
    old = len(part.blob)
    part._blob = blob
    part._content_type = "image/png"
    # scale inline size to the new aspect ratio
    w, h = swap_img.size
    EMU_PER_PX = 9525
    shape.width = int(w * EMU_PER_PX)
    shape.height = int(h * EMU_PER_PX)
    count += 1
    print("swapped image %s: old %dB -> new %dB, size %.2f x %.2f in" % (rId, old, len(blob), w / 200, h / 200))
doc.save(DOCX)
print("SAVED, images swapped:", count)
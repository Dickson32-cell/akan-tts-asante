# Akan TTS — EVAL_V3 SETTLE CELL (run in the SAME session, soon)
# Two actions, both explicit:
#  A) push GENTLE weights -> new repo Dickson32-cell/akan-twi-mms-gentle
#  B) RESTORE ORIGINAL 11940-step weights -> main repo akan-twi-mms (undo tonight's overwrite)
# Paste, press play. ~5 min. Must run BEFORE the Colab session dies (session-local dirs).
import os, glob, shutil, subprocess
from pathlib import Path
print("═══ AKAN TTS SETTLE CELL ═══", flush=True)

from huggingface_hub import HfApi, upload_file
api = HfApi()
me = api.whoami(); print("as:", me["name"])

GENTLE = "/content/akan-vits-finetuned"
OLD_DIR = "/content/model_OLD_11940"          # built by EVAL_V2b (Drive weights + base tokenizer)
assert glob.glob(os.path.join(GENTLE, "*.safetensors")), "gentle weights gone — session died? rerun training loader"
assert glob.glob(os.path.join(OLD_DIR, "*.safetensors")) or glob.glob("/content/drive/MyDrive/UG_TTS/akan-finetuned/model.safetensors"), "OLD weights source missing"
if not glob.glob(os.path.join(OLD_DIR, "*.safetensors")):
    os.makedirs(OLD_DIR, exist_ok=True)
    shutil.copy("/content/drive/MyDrive/UG_TTS/akan-finetuned/model.safetensors", os.path.join(OLD_DIR, "model.safetensors"))
    from huggingface_hub import hf_hub_download
    for f in ["vocab.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"]:
        try:
            shutil.copy(hf_hub_download("facebook/mms-tts-aka", f), os.path.join(OLD_DIR, f))
        except Exception:
            pass
print("weights sources ready")

WEIGHT_FILES = ["model.safetensors", "config.json", "tokenizer_config.json",
                "special_tokens_map.json", "vocab.json", "added_tokens.json"]
ARTIFACTS = ["loss_curve.png", "losses.csv", "convergence_summary.json", "train_akan.log"]

# A) gentle -> its own repo (never touches main)
print("\n--- A) gentle weights -> Dickson32-cell/akan-twi-mms-gentle ---")
api.create_repo("Dickson32-cell/akan-twi-mms-gentle", exist_ok=True, private=False)
for f in WEIGHT_FILES:
    p = os.path.join(GENTLE, f)
    if os.path.exists(p):
        upload_file(path_or_fileobj=p, path_in_repo=f, repo_id="Dickson32-cell/akan-twi-mms-gentle",
                    repo_type="model", commit_message="gentle run: LR 1e-4, 20 epochs, 7960 steps, seed 456 (CER 0.232 @0.667)")
        print("  uploaded:", f)
    else:
        print("  (absent, skipped)", f)
for f in ARTIFACTS:
    p = os.path.join(GENTLE, f)
    if not os.path.exists(p):
        p2 = "/content/" + f
        p = p2 if os.path.exists(p2) else None
    if p and os.path.exists(p):
        upload_file(path_or_fileobj=p, path_in_repo=f, repo_id="Dickson32-cell/akan-twi-mms-gentle",
                    repo_type="model")
        print("  artifact:", f)
print("gentle repo done: https://huggingface.co/Dickson32-cell/akan-twi-mms-gentle")

# B) restore ORIGINAL weights + tokenizer to the MAIN repo (undo the overwrite)
print("\n--- B) RESTORE original 11940 weights -> Dickson32-cell/akan-twi-mms ---")
for f in WEIGHT_FILES:
    p = os.path.join(OLD_DIR, f)
    if os.path.exists(p):
        upload_file(path_or_fileobj=p, path_in_repo=f, repo_id="Dickson32-cell/akan-twi-mms",
                    repo_type="model", commit_message="RESTORE original 11940-step weights (CER 0.324 @0.667)")
        print("  restored:", f)
    else:
        print("  MISSING in OLD dir (check), ", f)
# keep convergence artifacts on main too (they document tonight's run)
for f in ARTIFACTS:
    p = os.path.join(GENTLE, f)
    if not os.path.exists(p):
        p2 = "/content/" + f
        p = p2 if os.path.exists(p2) else None
    if p and os.path.exists(p):
        try:
            upload_file(path_or_fileobj=p, path_in_repo=f, repo_id="Dickson32-cell/akan-twi-mms", repo_type="model")
            print("  artifact kept:", f)
        except Exception as e:
            print("  artifact upload skip:", f, str(e)[:60])

print("\n✅ SETTLED: main repo = ORIGINAL 11940 (research baseline as reviewed);")
print("   gentle repo = tonight's improvement (CER 0.232, gap-to-base −39%).")
print("Send both lines above to Hermes for the card/report update.")
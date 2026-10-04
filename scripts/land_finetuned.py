"""Land the fine-tuned model + render everything (run AFTER training completes).

Steps: pull Dickson32-cell/akan-twi-mms from the Hub -> local 05_models/akan-finetuned,
render the 10 VETTED prompts through it -> 06_samples/final (the submission set),
render the 200-eval-subset comparison grid input, and print next actions.

Usage: python scripts/land_finetuned.py
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROJ = REPO.parent
HUB_ID = "Dickson32-cell/akan-twi-mms"
LOCAL = PROJ / "05_models" / "akan-finetuned"

def sh(cmd):
    print("+", cmd)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1200:]); print(r.stderr[-1200:])
    return r

def main():
    print("=== 1. Pull fine-tuned model from the Hub ===")
    py = sys.executable
    r = sh(f'"{py}" -c "from huggingface_hub import snapshot_download; '
           f"print(snapshot_download(repo_id='{HUB_ID}', local_dir=r'{LOCAL}'))\"")
    if r.returncode != 0 or not (LOCAL / "model.safetensors").exists() and not (LOCAL / "pytorch_model.bin").exists():
        raise SystemExit("model pull failed - is training finished and pushed?")

    print("=== 2. Render the 10 VETTED prompts with the FINE-TUNED model ===")
    sh(f'"{py}" "{REPO / "src" / "infer.py"}" --model "{LOCAL}" --prompts "{REPO / "scripts" / "demo_prompts.txt"}" '
       f'--outdir "{PROJ / "06_samples" / "final"}"')

    print("=== 3. Render the BASELINE equivalents (MOS before/after pair) ===")
    sh(f'"{py}" "{REPO / "src" / "infer.py"}" --model "{PROJ / "05_models" / "mms-tts-aka"}" '
       f'--prompts "{REPO / "scripts" / "demo_prompts.txt"}" --outdir "{PROJ / "06_samples" / "baseline_vetted"}')

    print("""
=== LANDING COMPLETE ===
Submission assets staged:
  model      : 05_models/akan-finetuned  (+ hub link lives at HF)
  final audio: 06_samples/final/demo_01..10.wav   (fine-tuned - THE deliverable)
  baseline   : 06_samples/baseline_vetted/demo_01..10.wav (before/after pair for MOS)
NEXT (agent or manual):
  python src/eval_asr.py --engine hf --asr-model <judge> --samples <proj>/06_samples/final
  run MOS panel (08_presentation/mos_panel/PROTOCOL.md)
  fill report sections 2-4 with real numbers, build .docx, assemble Drive folder
""")

if __name__ == "__main__":
    main()
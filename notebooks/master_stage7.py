# Akan TTS — STAGE 7+ (training → convergence → push)
# Fetched and run by the tiny loader cell in Colab. Always the current version.
# Assumes stages 1-6 already succeeded in this session (installs, login, data, base ckpt).
import os, sys, json, glob, subprocess, time
from pathlib import Path

def stage(n, t): print("\n" + "="*14 + f"  STAGE {n}: {t}  " + "="*14)

# ── config rewrite (safe every time; gentle settings + resume toggle live here) ──
stage(6, "Config")
cfg = {
  "project_name": "akan_twi_tts", "push_to_hub": True,
  "hub_model_id": "Dickson32-cell/akan-twi-mms",
  "report_to": ["tensorboard"], "logging_strategy": "steps", "logging_steps": 20,
  "overwrite_output_dir": False,
  "output_dir": "/content/akan-vits-finetuned",
  "dataset_name": "/content/tts_data", "dataset_config_name": None,
  "audio_column_name": "audio", "text_column_name": "text",
  "train_split_name": "train", "eval_split_name": "eval",
  "speaker_id_column_name": "speaker_id", "filter_on_speaker_id": 0,
  "override_speaker_embeddings": False,
  "full_generation_sample_text": "Mema wo akye, me nuanom. Wo ho te s\u025bn?",
  "max_duration_in_seconds": 12.0, "min_duration_in_seconds": 2.0, "max_tokens_length": 500,
  "model_name_or_path": "/content/aka_train_base",
  "preprocessing_num_workers": 1,
  "do_train": True, "num_train_epochs": 20,
  "gradient_accumulation_steps": 1, "per_device_train_batch_size": 16,
  "learning_rate": 1e-4, "lr_scheduler_type": "cosine", "warmup_ratio": 0.02,
  "adam_beta1": 0.8, "adam_beta2": 0.99, "weight_decay": 0.01, "group_by_length": False,
  "do_eval": True, "eval_steps": 500, "per_device_eval_batch_size": 8,
  "max_eval_samples": 25, "do_step_schedule_per_epoch": False,
  "save_steps": 1000, "save_total_limit": 3,
  "weight_disc": 3, "weight_fmaps": 1, "weight_gen": 1, "weight_kl": 1.5,
  "weight_duration": 1, "weight_mel": 35, "fp16": True, "seed": 456
}
resume = bool(glob.glob("/content/akan-vits-finetuned/checkpoint-*"))
if resume:
    cfg["resume_from_checkpoint"] = True
    print("checkpoints found \u2192 RESUME (via config)")
else:
    cfg.pop("resume_from_checkpoint", None)
    print("no checkpoints \u2192 FRESH RUN")
Path("/content/finetune_akan.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print("config written: LR 1e-4, 20 epochs, repo Dickson32-cell/akan-twi-mms")

# ── training ──
stage(7, "Training")
LOG = "/content/train_akan.log"
RUNV = "run_vits_finetuning.py"
script = "/content/finetune-hf-vits/run_vits_finetuning.py"
if not Path(script).exists():
    script = "/content/finetune-hf-vits/" + RUNV
    if not Path(script).exists():
        raise SystemExit("harness not found \u2014 run the master cell (stage 2) first")
# HARNESS RULE (verified at source, line 530): config json must be the ONLY argument.
cmd = ["python", "-m", "accelerate.commands.launch", script, "/content/finetune_akan.json"]
print("launch args:", cmd)
log_f = open(LOG, "a")
proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, bufsize=1,
                        universal_newlines=True, cwd=os.path.dirname(script))
Path("/content/train_pid.txt").write_text(str(proc.pid))
print("launched pid:", proc.pid, "\u2014 losses will start appearing after a few minutes")

last = 0
while proc.poll() is None:
    time.sleep(25)
    try:
        lines = Path(LOG).read_text(errors="replace").splitlines()
    except FileNotFoundError:
        continue
    for l in lines[last:]:
        if any(k in l.lower() for k in ("loss", "eval_", "error", "traceback", "epoch", "it/s")):
            print(l[:180])
    last = len(lines)
log_f.close()
n_lines = sum(1 for _ in open(LOG, "r", errors="replace"))
print("\n=== TRAINING EXITED rc=%s | log lines: %d ===" % (proc.returncode, n_lines))
if proc.returncode != 0:
    print("TRAINING FAILED \u2014 last lines:")
    for l in Path(LOG).read_text(errors="replace").splitlines()[-12:]:
        print("   ", l[:180])
    raise SystemExit("Send the lines above to Hermes.")

# ── convergence report ──
stage(8, "Convergence report")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
cks = sorted(glob.glob("/content/akan-vits-finetuned/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
state_file = None
if cks and Path(cks[-1], "trainer_state.json").exists():
    state_file = os.path.join(cks[-1], "trainer_state.json")
if not state_file:
    alt = sorted(glob.glob("/content/drive/MyDrive/**/trainer_state.json", recursive=True))
    if alt: state_file = alt[0]
if not state_file:
    print("convergence report skipped \u2014 no checkpoints found")
else:
    st = json.load(open(state_file))
    hist = st.get("log_history", [])
    train = [(h["step"], h["loss"]) for h in hist if "loss" in h]
    evals = [(h["step"], v) for h in hist for k, v in h.items()
             if k.startswith("eval_") and isinstance(v, (int, float)) and not k.endswith("runtime")]
    print("steps trained:", st.get("global_step"), "| train pts:", len(train), "| eval pts:", len(evals))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if train: xs, ys = zip(*train); ax.plot(xs, ys, lw=1.5, label="train loss")
    if evals:
        k0 = "eval_loss" if any(k == "eval_loss" for _, k in evals) else evals[0][1]
        ax.plot([s for s, k in evals if k == k0], [v for s, k in evals if k == k0], "o-", label=k0)
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("Akan TTS fine-tune \u2014 convergence")
    fig.tight_layout()
    fig.savefig("/content/akan-vits-finetuned/loss_curve.png", dpi=130)
    verdict = "CHECK CURVE MANUALLY"
    if len(train) >= 5:
        n5 = max(1, len(train) // 10)
        first10 = sum(y for _, y in train[:n5]) / n5
        last10 = sum(y for _, y in train[-n5:]) / n5
        drop = first10 - last10
        tail = [y for _, y in train[-max(1, len(hist) // 5):]]
        flat = (max(tail) - min(tail)) if tail else 0.0
        if drop <= 0 and flat > 0.2 * first10:
            verdict = "NOT CONVERGING \u2014 loss rising/oscillating"
        elif drop <= 0.02 * first10 and flat < 0.02 * last10:
            verdict = "CONVERGED \u2014 plateau reached"
        elif drop > 0:
            verdict = "CONVERGING \u2014 total drop %.4f (%.1f%%); tail osc \u00b1%.4f" % (drop, drop / first10 * 100.0, flat)
    print("VERDICT:", verdict)
    json.dump({"global_step": st.get("global_step"), "verdict": verdict,
               "state_file": state_file, "n_train_points": len(train)},
              open("/content/akan-vits-finetuned/convergence_summary.json", "w"), indent=2)

# ── save + push ──
stage(9, "Save to Drive + push artifacts to Hugging Face")
os.makedirs("/content/drive/MyDrive/UG_TTS", exist_ok=True)
subprocess.run("zip -qr '/content/drive/MyDrive/UG_TTS/akan_tts_ckpt_new.zip' /content/akan-vits-finetuned || true",
               shell=True)
from huggingface_hub import upload_file
for f in ["/content/akan-vits-finetuned/loss_curve.png",
          "/content/akan-vits-finetuned/convergence_summary.json",
          "/content/train_akan.log"]:
    if os.path.exists(f):
        try:
            upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f),
                        repo_id="Dickson32-cell/akan-twi-mms", repo_type="model")
            print("uploaded:", os.path.basename(f))
        except Exception as e:
            print("upload failed (keep the Drive copy):", f, str(e)[:120])
print("\n\u2705 DONE \u2014 artifacts on the Hub: https://huggingface.co/Dickson32-cell/akan-twi-mms")
print("Send the TRAINING EXITED rc= line and the VERDICT line to Hermes.")
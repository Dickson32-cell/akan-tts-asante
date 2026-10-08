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

# --- HARNESS PATCHES (verbatim from the verified TRAIN-ONE-CELL run) ---
stage("6b", "Patching harness (3 verified fixes)")
_h = '/content/finetune-hf-vits/run_vits_finetuning.py'
_s = open(_h, encoding='utf-8').read()
if 'load_from_disk(data_args.dataset_name)' not in _s:
    import re as _re
    _pat_train = r'raw_datasets\[.train.\] = load_dataset\((?:.|\n)*?\)\n'
    _s = _re.sub(_pat_train, 'raw_datasets["train"] = load_from_disk(data_args.dataset_name)[data_args.train_split_name]\n', _s, count=1)
    _pat_eval = r'raw_datasets\[.eval.\] = load_dataset\((?:.|\n)*?\)\n'
    _s = _re.sub(_pat_eval, 'raw_datasets["eval"] = load_from_disk(data_args.dataset_name)[data_args.eval_split_name]\n', _s, count=1)
    if 'from datasets import DatasetDict, load_dataset' in _s:
        _s = _s.replace('from datasets import DatasetDict, load_dataset',
                        'from datasets import DatasetDict, load_dataset, load_from_disk')
    open(_h, 'w', encoding='utf-8').write(_s)
    print('harness PATCHED: load_from_disk')
_s = open(_h, encoding='utf-8').read()
_slice_old = 'batch[model_input_name] = string_inputs.get("input_ids")[: max_tokens_length + 1]'
if _slice_old in _s:
    _s = _s.replace(_slice_old,
        'batch[model_input_name] = string_inputs.get("input_ids")[: int(max_tokens_length) + 1]')
    open(_h, 'w', encoding='utf-8').write(_s)
    print('harness PATCHED: float-slice int cast')
else:
    print('harness slice already patched')
_plot = '/content/finetune-hf-vits/utils/plot.py'
_ps = open(_plot, encoding='utf-8').read()
if 'tostring_rgb' in _ps:
    _ps = _ps.replace('data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)',
                      'data = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()')
    _ps = _ps.replace('np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).astype(np.uint8)',
                      'np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()')
    open(_plot, 'w', encoding='utf-8').write(_ps)
    print('harness PATCHED: modern matplotlib buffer_rgba')
else:
    print('plot.py already modern')

# PATCH 4 (env drift Oct 2026): datasets 3.6 features/video.py top-level
# `from torchvision.io import VideoReader`; torchvision >= 0.20 removed the class,
# so `import datasets` crashes before training starts (tonight's VideoReader error).
import datasets as _ds
_dsdir = os.path.dirname(_ds.__file__)
_vid = os.path.join(_dsdir, "features", "video.py")
if os.path.exists(_vid) and "from torchvision.io import VideoReader" in open(_vid, encoding="utf-8", errors="replace").read():
    _vs = open(_vid, encoding="utf-8", errors="replace").read()
    _vs = _vs.replace("from torchvision.io import VideoReader",
                      "try:\n    from torchvision.io import VideoReader\nexcept ImportError:\n    VideoReader = None", 1)
    open(_vid, "w", encoding="utf-8").write(_vs)
    print("harness PATCHED: torchvision VideoReader import made optional (env-drift fix)")
else:
    print("video.py already free of top-level VideoReader import (fixed earlier)")
import importlib
importlib.reload_path if False else None
for _m in list(sys.modules):
    if _m == "datasets" or _m.startswith("datasets."):
        del sys.modules[_m]
import importlib as _il
_ds2 = _il.import_module("datasets")
print("datasets re-imports OK after video.py patch")

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

# ── convergence report (tfevents — the harness's real log; not trainer_state.json) ──
stage(8, "Convergence report from tfevents")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ev_files = sorted(glob.glob("/content/akan-vits-finetuned/**/events.out.tfevents.*", recursive=True))
if not ev_files:
    print("no tfevents yet — training produced none (check stage 7 output)")
else:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(os.path.dirname(ev_files[-1]))
    acc.Reload()
    tags = [t for t in acc.Tags().get("scalars", [])]
    print("tfevent scalar tags:", tags)
    CURVES = ["train_summed_losses", "train_loss_mel", "train_loss_kl",
              "train_loss_gen", "train_loss_disc", "train_loss_duration", "train_loss_fmaps"]
    fig, ax = plt.subplots(figsize=(10, 5))
    all_curves = []
    for tag in [t for t in CURVES if t in tags]:
        pts = [(e.step, e.value) for e in acc.Scalars(tag)]
        all_curves.append((tag, pts))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.3, label=tag.replace("train_loss_", ""))
    try:
        ax.set_yscale("log")
    except Exception:
        pass
    ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("Akan TTS fine-tune — convergence (tfevents)")
    fig.tight_layout()
    fig.savefig("/content/akan-vits-finetuned/loss_curve.png", dpi=130)
    import csv as _csv
    with open("/content/akan-vits-finetuned/losses.csv", "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["step"] + [t for t, _ in all_curves])
        dicts = [{p[0]: p[1] for p in pts} for _, pts in all_curves]
        for st_ in sorted({p[0] for _, pts in all_curves for p in pts}):
            w.writerow([st_] + [d.get(st_) for d in dicts])
    print("saved losses.csv + loss_curve.png")
    main_tag = "train_summed_losses" if "train_summed_losses" in tags else (tags[0] if tags else None)
    verdict = "CHECK CURVE MANUALLY"
    if main_tag:
        vals = [p[1] for p in [(e.step, e.value) for e in acc.Scalars(main_tag)]]
        if len(vals) >= 20:
            k = max(2, len(vals) // 10)
            first = sum(vals[:k]) / k
            last = sum(vals[-k:]) / k
            drop = first - last
            tailspan = max(vals[-k:]) - min(vals[-k:])
            if tailspan > 0.35 * max(first, 1e-9):
                verdict = "NOT STABLE YET — high oscillation"
            elif drop <= 0.02 * first:
                verdict = "CONVERGED — plateau reached (drop %.4f)" % drop
            else:
                verdict = "CONVERGING — drop %.4f (%.1f%%)" % (drop, 100 * drop / max(first, 1e-9))
    print("VERDICT:", verdict)
    json.dump({"verdict": verdict, "tags": tags, "event_dir": os.path.dirname(ev_files[-1])},
              open("/content/akan-vits-finetuned/convergence_summary.json", "w"), indent=2)

# ── save + push ──
stage(9, "Save to Drive + push artifacts to Hugging Face")
os.makedirs("/content/drive/MyDrive/UG_TTS", exist_ok=True)
subprocess.run("zip -qr '/content/drive/MyDrive/UG_TTS/akan_tts_ckpt_new.zip' /content/akan-vits-finetuned || true",
               shell=True)
from huggingface_hub import upload_file
for f in ["/content/akan-vits-finetuned/loss_curve.png",
          "/content/akan-vits-finetuned/convergence_summary.json",
          "/content/train_akan.log",
          "/content/akan-vits-finetuned/losses.csv"]:
    if os.path.exists(f):
        try:
            upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f),
                        repo_id="Dickson32-cell/akan-twi-mms", repo_type="model")
            print("uploaded:", os.path.basename(f))
        except Exception as e:
            print("upload failed (keep the Drive copy):", f, str(e)[:120])
print("\n\u2705 DONE \u2014 artifacts on the Hub: https://huggingface.co/Dickson32-cell/akan-twi-mms")
print("Send the TRAINING EXITED rc= line and the VERDICT line to Hermes.")
# Akan TTS - STAGE 6-9 FREEZE RUN (patches -> train with FROZEN text-encoder+flow -> convergence -> isolated push)
# Key differences vs master_stage7 (gentle run):
#   1. cfg.hub_model_id = Dickson32-cell/akan-twi-mms-freeze  (NEW OWN repo; main + gentle repos are untouchable)
#   2. LR 5e-5, 12 epochs, seed 456 (same family for comparability)
#   3. AFTER harness patches: insert a freeze block into run_vits_finetuning.py that sets
#      requires_grad=False for model.text_encoder and model.flow parameters before training starts
#      (official staged fine-tuning practice for small data - protects the base phonetics).
#   4. resume: only from THIS run's checkpoints (output_dir = /content/akan-vits-finetuned-freeze)
# Assumes stages 1-5 already succeeded via freeze_stage1.py.
import os, sys, json, glob, subprocess, time, re as _re
from pathlib import Path

def stage(n, t): print("\n" + "="*14 + f"  STAGE {n}: {t}  " + "="*14, flush=True)

# ── config ──
stage(6, "Freeze-run config")
cfg = {
  "project_name": "akan_twi_tts_freeze", "push_to_hub": True,
  "hub_model_id": "Dickson32-cell/akan-twi-mms-freeze",
  "report_to": ["tensorboard"], "logging_strategy": "steps", "logging_steps": 20,
  "overwrite_output_dir": False,
  "output_dir": "/content/akan-vits-finetuned-freeze",
  "dataset_name": "/content/tts_data", "dataset_config_name": None,
  "audio_column_name": "audio", "text_column_name": "text",
  "train_split_name": "train", "eval_split_name": "eval",
  "speaker_id_column_name": "speaker_id", "filter_on_speaker_id": 0,
  "override_speaker_embeddings": False,
  "full_generation_sample_text": "Mema wo akye, me nuanom. Wo ho te s\u025bn?",
  "max_duration_in_seconds": 12.0, "min_duration_in_seconds": 2.0, "max_tokens_length": 500,
  "model_name_or_path": "/content/aka_train_base",
  "preprocessing_num_workers": 1,
  "do_train": True, "num_train_epochs": 12,
  "gradient_accumulation_steps": 1, "per_device_train_batch_size": 16,
  "learning_rate": 5e-5, "lr_scheduler_type": "cosine", "warmup_ratio": 0.02,
  "adam_beta1": 0.8, "adam_beta2": 0.99, "weight_decay": 0.01, "group_by_length": False,
  "do_eval": True, "eval_steps": 500, "per_device_eval_batch_size": 8,
  "max_eval_samples": 25, "do_step_schedule_per_epoch": False,
  "save_steps": 1000, "save_total_limit": 3,
  "weight_disc": 3, "weight_fmaps": 1, "weight_gen": 1, "weight_kl": 1.5,
  "weight_duration": 1, "weight_mel": 35, "fp16": True, "seed": 456
}
resume = bool(glob.glob("/content/akan-vits-finetuned-freeze/checkpoint-*"))
if resume:
    cfg["resume_from_checkpoint"] = "latest"
    print("freeze checkpoints found \u2192 RESUME (latest)")
else:
    cfg.pop("resume_from_checkpoint", None)
    print("no freeze checkpoints \u2192 FRESH RUN")
Path("/content/finetune_akan_freeze.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print("freeze config written: LR 5e-5, 12 epochs, hub Dickson32-cell/akan-twi-mms-freeze")

# ── harness patches (IDENTICAL to the verified gentle flow) ──
stage("6b", "Patching harness (4 verified fixes)")
_h = '/content/finetune-hf-vits/run_vits_finetuning.py'
_s = open(_h, encoding='utf-8').read()
if 'load_from_disk(data_args.dataset_name)' not in _s:
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
    open(_plot, 'w', encoding='utf-8').write(_ps)
    print('harness PATCHED: modern matplotlib buffer_rgba')
else:
    print('plot.py already modern')

# PATCH 4: datasets video.py optional import (compile-checked splice, idempotent)
_vid_candidates = sorted(
    glob.glob("/usr/local/lib/python3*/dist-packages/datasets/features/video.py")
    + glob.glob("/usr/lib/python3*/site-packages/datasets/features/video.py")
)
_vid = _vid_candidates[0] if _vid_candidates else None
if _vid is None:
    print("video.py not found - skip patch 4")
else:
    _cur = open(_vid, encoding="utf-8", errors="replace").read()
    if "except ImportError" in _cur and "VideoReader = None" in _cur:
        print("video.py already patched (idempotent skip)")
    else:
        _out, _done = [], False
        for _l in _cur.splitlines(keepends=True):
            if not _done and _l.strip() == "from torchvision.io import VideoReader":
                _ind = _l[: len(_l) - len(_l.lstrip())]
                _out.append(_ind + "try:")
                _out.append(_ind + "    from torchvision.io import VideoReader")
                _out.append(_ind + "except ImportError:")
                _out.append(_ind + "    VideoReader = None")
                _done = True
            else:
                _out.append(_l)
        assert _done, "VideoReader import line not found"
        _new = "\n".join([l.rstrip() for l in "".join(_out).splitlines()])
        compile(_new, _vid, "exec")
        open(_vid, "w", encoding="utf-8").write(_new + "\n")
        print("harness PATCHED: video.py optional import (compile-verified)")

# PATCH 5 (FREEZE-RUN ONLY): insert freeze block after model construction
# The harness builds the model somewhere named `model` inside main(); we hook AFTER the
# line that creates the optimizer is NOT reliable - instead we hook the documented freeze point:
# freeze text_encoder + flow params right before Trainer creation via a tiny string-insert.
_freeze_marker = 'model_name_or_path'
if '# FREEZE-RUN INSERT' not in _s:
    _insert = '''
# FREEZE-RUN INSERT (this repo run only): protect base phonetics for small-data fine-tune
_frozen, _kept = 0, 0
for _n, _p in model.named_parameters():
    if _n.startswith("text_encoder.") or _n.startswith("flow."):
        _p.requires_grad = False; _frozen += 1
    else:
        _kept += 1
print(f"[FREEZE] text_encoder+flow frozen: {_frozen} params frozen / {_kept} trainable")
del _frozen, _kept
'''
    # insert right after the first "model.eval()" line in the harness (all VITS harness versions have one)
    m = _re.search(r'\n(\s*)model\.eval\(\)', _s)
    assert m, "model.eval() not found in harness - inspect the harness for the eval() line"
    insert_at = m.end()
    _s = _s[:insert_at] + "\n" + _insert + _s[insert_at:]
    open(_h, 'w', encoding='utf-8').write(_s)
    compile(_s, _h, "exec")
    print("harness PATCHED: FREEZE insert (compile-verified) - text_encoder + flow frozen")
else:
    print("freeze insert already present")

# ── training ──
stage(7, "Training (freeze run)")
LOG = "/content/train_akan_freeze.log"
script = "/content/finetune-hf-vits/run_vits_finetuning.py"
if not Path(script).exists():
    raise SystemExit("harness not found - run freeze_stage1.py (stages 1-5) first")
cmd = ["python", "-m", "accelerate.commands.launch", script, "/content/finetune_akan_freeze.json"]
print("launch args:", cmd)
log_f = open(LOG, "a")
proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, bufsize=1,
                        universal_newlines=True, cwd=os.path.dirname(script))
Path("/content/train_freeze_pid.txt").write_text(str(proc.pid))
print("launched pid:", proc.pid, "- losses will start appearing after a few minutes")

last = 0
while proc.poll() is None:
    time.sleep(25)
    try:
        lines = Path(LOG).read_text(errors="replace").splitlines()
    except FileNotFoundError:
        continue
    for l in lines[last:]:
        if any(k in l.lower() for k in ("loss", "eval_", "error", "traceback", "epoch", "it/s", "freeze")):
            print(l[:180])
    last = len(lines)
log_f.close()
n_lines = sum(1 for _ in open(LOG, "r", errors="replace"))
print("\n=== TRAINING EXITED rc=%s | log lines: %d ===" % (proc.returncode, n_lines))
if proc.returncode != 0:
    print("TRAINING FAILED - last lines:")
    for l in Path(LOG).read_text(errors="replace").splitlines()[-12:]:
        print("   ", l[:180])
    raise SystemExit("Review the lines above.")
# confirm the freeze actually engaged
freeze_line = [l for l in Path(LOG).read_text(errors="replace").splitlines() if "[FREEZE]" in l]
print(freeze_line[-1] if freeze_line else "WARNING: no FREEZE line in log - check insert placement")

# ── convergence report (tfevents) ──
stage(8, "Convergence report from tfevents")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ev_files = sorted(glob.glob("/content/akan-vits-finetuned-freeze/**/events.out.tfevents.*", recursive=True))
if not ev_files:
    print("no tfevents yet - nothing to plot")
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
    ax.set_title("Akan TTS FREEZE fine-tune - convergence (tfevents)")
    fig.tight_layout()
    fig.savefig("/content/akan-vits-finetuned-freeze/loss_curve.png", dpi=130)
    import csv as _csv
    with open("/content/akan-vits-finetuned-freeze/losses.csv", "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["step"] + [t for t, _ in all_curves])
        dicts = [{p[0]: p[1] for p in pts} for _, pts in all_curves]
        for st_ in sorted({p[0] for _, pts in all_curves for p in pts}):
            w.writerow([st_] + [d.get(st_) for d in dicts])
    print("saved losses.csv + loss_curve.png")
    main_tag = "train_summed_losses" if "train_summed_losses" in tags else (tags[0] if tags else None)
    if main_tag:
        vals = [p[1] for p in [(e.step, e.value) for e in acc.Scalars(main_tag)]]
        if len(vals) >= 20:
            k = max(2, len(vals) // 10)
            first = sum(vals[:k]) / k
            last_v = sum(vals[-k:]) / k
            drop = first - last_v
            tailspan = max(vals[-k:]) - min(vals[-k:])
            if tailspan > 0.35 * max(first, 1e-9):
                verdict = "NOT STABLE YET - high oscillation"
            elif drop <= 0.02 * first:
                verdict = "CONVERGED - plateau reached (drop %.4f)" % drop
            else:
                verdict = "CONVERGING - drop %.4f (%.1f%%)" % (drop, 100 * drop / max(first, 1e-9))
            print("VERDICT:", verdict)
    json.dump({"verdict": locals().get("verdict", "CHECK MANUALLY"), "tags": tags},
              open("/content/akan-vits-finetuned-freeze/convergence_summary.json", "w"), indent=2)

# ── save + ISOLATED push (NEW repo only) ──
stage(9, "Save to Drive + push FREEZE artifacts to its OWN repo")
os.makedirs("/content/drive/MyDrive/UG_TTS", exist_ok=True)
subprocess.run("zip -qr '/content/drive/MyDrive/UG_TTS/akan_tts_ckpt_freeze.zip' /content/akan-vits-finetuned-freeze || true",
               shell=True)
# create the repo if missing, then push every artifact + weights
from huggingface_hub import HfApi, upload_folder, upload_file
api = HfApi()
try:
    api.create_repo(repo_id="Dickson32-cell/akan-twi-mms-freeze", repo_type="model", exist_ok=True)
    print("repo ready: Dickson32-cell/akan-twi-mms-freeze")
except Exception as e:
    print("create_repo:", str(e)[:120])
outdir = "/content/akan-vits-finetuned-freeze"
try:
    upload_folder(folder_path=outdir, repo_id="Dickson32-cell/akan-twi-mms-freeze", repo_type="model",
                  allow_patterns=["*.json", "*.txt", "*.safetensors", "*.png", "*.csv", "*.md"])
    print("weights+config uploaded to FREEZE repo")
except Exception as e:
    print("upload_folder failed:", str(e)[:150])
for f in ["loss_curve.png", "convergence_summary.json", "losses.csv"]:
    p = os.path.join(outdir, f)
    if os.path.exists(p):
        try:
            upload_file(path_or_fileobj=p, path_in_repo=f,
                        repo_id="Dickson32-cell/akan-twi-mms-freeze", repo_type="model")
        except Exception as e:
            print("upload failed:", f, str(e)[:120])
if os.path.exists(LOG):
    try:
        upload_file(path_or_fileobj=LOG, path_in_repo="train_akan_freeze.log",
                    repo_id="Dickson32-cell/akan-twi-mms-freeze", repo_type="model")
    except Exception as e:
        print("log upload failed:", str(e)[:120])
print("\n\u2705 DONE - FREEZE weights live ONLY at: https://huggingface.co/Dickson32-cell/akan-twi-mms-freeze")
print("(main akan-twi-mms + gentle repos were never touched by this run)")
print("the TRAINING EXITED rc= line, the [FREEZE] line, and the VERDICT line summarize the run.")
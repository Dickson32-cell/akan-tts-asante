# Akan TTS - FINAL CELL v2 (self-contained: heal-check, truncating monitor, true-error report)
# Run via the 2-line URL loader. Works in ANY kernel state (fresh or poisoned) because
# nothing ever imports the datasets package in-kernel — subprocess probes only.
import os, sys, json, glob, subprocess, time, urllib.request
from pathlib import Path

print("═══ AKAN TTS FINAL CELL v2 ═══", flush=True)

def stage(n, t): print("\n" + "="*14 + f"  STAGE {n}: {t}  " + "="*14, flush=True)

def shell(cmd, must=True, cwd=None, timeout=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd, timeout=timeout)
    if r.returncode != 0 and must:
        print("COMMAND FAILED:", cmd)
        print((r.stdout or "")[-1200:]); print((r.stderr or "")[-1200:])
        print(">>> paste the lines above to Hermes in Telegram, then re-run this cell.")
        return r
    return r

stage(0, "Kernel-state + file heal-check (defensive, subprocess-only)")
# heal-on-contact for the video.py wound (compile-gated; idempotent)
_vids = glob.glob("/usr/local/lib/python3*/dist-packages/datasets/features/video.py")
if _vids:
    _vid = _vids[0]
    _src = open(_vid, encoding="utf-8", errors="replace").read()
    def _ok(t, v=_vid):
        try: compile(t, v, "exec"); return True
        except Exception: return False
    if _ok(_src):
        print("datasets file state: compiles OK (healed or healthy)")
    else:
        print("datasets video.py does not compile - healing TYPE_CHECKING span...")
        _lines = _src.splitlines(keepends=True)
        try:
            _ti = next(i for i, l in enumerate(_lines) if l.strip() == "if TYPE_CHECKING:")
            _fi = next(i for i in range(_ti, min(_ti + 14, len(_lines)))
                       if "from .features import FeatureType" in _lines[i])
            _canon = ["if TYPE_CHECKING:\n", "    try:\n",
                      "        from torchvision.io import VideoReader\n",
                      "    except ImportError:\n", "        VideoReader = None\n",
                      "\n", "    from .features import FeatureType\n"]
            _new = "".join(_lines[:_ti] + _canon + _lines[_fi + 1:])
        except StopIteration:
            _new = _src.replace("from torchvision.io import VideoReader",
                                "try:\n    from torchvision.io import VideoReader\nexcept ImportError:\n    VideoReader = None")
        assert _ok(_new), "heal failed - paste this output to Hermes"
        open(_vid, "w", encoding="utf-8").write(_new)
        print("HEALED + compile-verified")
_r = shell('python -c "import datasets; print(\'REIMPORT-OK\', datasets.__version__)"', must=False)
_p = (( _r.stdout or _r.stderr) or "").strip().splitlines()
print("fresh-python probe:", (_p[-1] if _p else "(none)"))
if _r.returncode != 0:
    print("STILL POISONED for fresh python - the 25-line tail follows this message; paste it whole to Hermes.")
    _tail = (_r.stdout or "") + (_r.stderr or "")
    print("\n".join(_tail.splitlines()[-25:]))

# stages 1-3 (fast, idempotent)
stage(1, "GPU")
import torch
if not torch.cuda.is_available():
    print("NO GPU: Runtime > Change runtime type > T4 GPU > save > run again")
    raise SystemExit('NO GPU - rerun after GPU set')
print("GPU OK:", torch.cuda.get_device_name(0))

stage(2, "Versions (metadata) + harness")
import importlib.metadata as md
_vt, _vd = md.version("transformers"), md.version("datasets")
print("pins:", _vt, "| datasets", _vd, "| harness:",
      Path("/content/finetune-hf-vits/run_vits_finetuning.py").exists())
assert Path("/content/finetune-hf-vits/run_vits_finetuning.py").exists(), "harness missing - rerun the long loader first"

stage(3, "HF login")
from huggingface_hub import HfApi
_a = HfApi()
try:
    print("logged in as:", _a.whoami()["name"])
except Exception:
    from huggingface_hub import notebook_login
    notebook_login()

stage(4, "Corpus + base present?")
assert Path("/content/tts_data").exists(), "tts_data missing - rerun the long loader first"
assert glob.glob("/content/aka_train_base/*.safetensors"), "aka_train_base missing - rerun the long loader first"
print("tts_data OK | base OK")



stage(6, "Config (gentle run, repo fixed)")
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
_resume = bool(glob.glob("/content/akan-vits-finetuned/checkpoint-*"))
if _resume:
    cfg["resume_from_checkpoint"] = True
    print("checkpoints present -> RESUME via config")
else:
    cfg.pop("resume_from_checkpoint", None)
    print("no checkpoints -> FRESH RUN")
Path("/content/finetune_akan.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

stage(6.5, "Harness patch verification (4 fixes present?)")
_H = "/content/finetune-hf-vits/run_vits_finetuning.py"
_hs = open(_H, encoding="utf-8").read()
print("load_from_disk patched:", "load_from_disk(data_args.dataset_name)" in _hs)
print("int-cast patched:", "int(max_tokens_length)" in _hs)
if "load_from_disk(data_args.dataset_name)" not in _hs:
    print("PATCHES MISSING - rerun the long loader first (it applies all 4)")
    print(">>> paste this line to Hermes")

stage(7, "Training (TRUNCATING monitor - only THIS run's output shows)")
LOG = "/content/train_akan.log"
MARK = len(Path(LOG).read_text(errors="replace").splitlines()) if Path(LOG).exists() else 0
cmd = ["python", "-m", "accelerate.commands.launch",
       "/content/finetune-hf-vits/run_vits_finetuning.py",
       "/content/finetune_akan.json"]
print("launch args:", cmd)
with open(LOG, "a") as lf:
    lf.write("\n@@@@@@@@ NEW-ATTEMPT-MARKER %d @@@@@@@@\n" % time.time())
proc = subprocess.Popen(cmd, stdout=open(LOG, "a"), stderr=subprocess.STDOUT,
                        bufsize=1, universal_newlines=True,
                        cwd="/content/finetune-hf-vits")
Path("/content/train_pid.txt").write_text(str(proc.pid))
print("launched pid:", proc.pid, "- losses appear in a few minutes; this runs for hours")

_last = MARK
_seen_marker = False
while proc.poll() is None:
    time.sleep(30)
    try:
        _lines = Path(LOG).read_text(errors="replace").splitlines()
    except FileNotFoundError:
        continue
    if not _seen_marker:
        _skip_to = next((i for i, l in enumerate(_lines) if l.startswith("@@@@@@@@ NEW-ATTEMPT-MARKER")), None)
        if _skip_to is None:
            continue
        _last = _skip_to + 1
        _seen_marker = True
    for l in _lines[_last:]:
        _ll = l.lower()
        if any(k in _ll for k in ("loss", "eval_", "epoch", "it/s", "error", "traceback")):
            print(l[:170], flush=True)
    _last = len(_lines)
rc = proc.returncode
print("\n=== TRAINING EXITED rc=%s ===" % rc, flush=True)
if rc != 0:
    _fresh = Path(LOG).read_text(errors="replace").splitlines()
    _midx = [i for i, l in enumerate(_fresh) if l.startswith("@@@@@@@@ NEW-ATTEMPT-MARKER")]
    _seg = _fresh[(_midx[-1] + 1) if _midx else max(0, len(_fresh) - 40):]
    print("THIS ATTEMPT's last 25 lines (the true error):")
    for l in _seg[-25:]:
        print("   ", l[:170])
    print(">>> paste ALL lines from 'THIS ATTEMPT' to Hermes")
    raise SystemExit('NO GPU - set T4 GPU in Runtime menu, then re-run this cell')

stage(8, "Convergence report (tfevents + CSV)")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
_ev = sorted(glob.glob("/content/akan-vits-finetuned/**/events.out.tfevents.*", recursive=True))
if not _ev:
    print("no tfevents found - check training output above")
else:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        acc = EventAccumulator(os.path.dirname(_ev[-1])); acc.Reload()
        tags = acc.Tags().get("scalars", [])
        print("tags:", tags)
        CURVES = ["train_summed_losses", "train_loss_mel", "train_loss_kl",
                  "train_loss_gen", "train_loss_disc", "train_loss_duration", "train_loss_fmaps"]
        _use = [t for t in CURVES if t in tags] or tags[:5]
        fig, ax = plt.subplots(figsize=(10, 5))
        for t in _use:
            pts = [(e.step, e.value) for e in acc.Scalars(t)]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.2, label=t.replace("train_loss_", ""))
        try: ax.set_yscale("log")
        except Exception: pass
        ax.set_xlabel("step"); ax.grid(alpha=.3); ax.legend()
        ax.set_title("Akan TTS fine-tune - convergence")
        fig.tight_layout(); fig.savefig("/content/akan-vits-finetuned/loss_curve.png", dpi=130)
        import csv
        with open("/content/akan-vits-finetuned/losses.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(["step"] + _use)
            _d = [dict((e.step, e.value) for e in acc.Scalars(t)) for t in _use]
            _maps = [dict(p) for p in _d]
            for s_ in sorted({p[0] for p in _maps[0]} if _maps else []):
                w.writerow([s_] + [m.get(s_) for m in _maps])
        _t0 = _use[0]
        vals = [e.value for e in acc.Scalars(_t0)]
        verdict = "CHECK CURVE MANUALLY"
        if len(vals) >= 20:
            k = max(2, len(vals)//10)
            first, last = sum(vals[:k])/k, sum(vals[-k:])/k
            drop = first - last
            span = max(vals[-k:]) - min(vals[-k:])
            if span > 0.35 * max(first, 1e-9): verdict = "NOT STABLE YET - oscillating"
            elif drop <= 0.02 * first: verdict = "CONVERGED - plateau (drop %.4f)" % drop
            else: verdict = "CONVERGING - drop %.4f (%.1f%%)" % (drop, 100*drop/max(first, 1e-9))
        print("VERDICT:", verdict)
        json.dump({"verdict": verdict}, open("/content/akan-vits-finetuned/convergence_summary.json", "w"), indent=2)
    except Exception:
        import traceback; traceback.print_exc()
        print("(convergence parse failed - send the trace above)")

stage(9, "Save to Drive + push artifacts")
os.makedirs("/content/drive/MyDrive/UG_TTS", exist_ok=True) if os.path.exists("/content/drive/MyDrive") else None
shell("zip -qr '/content/drive/MyDrive/UG_TTS/akan_tts_ckpt_new.zip' /content/akan-vits-finetuned || true", must=False)
from huggingface_hub import upload_file
for f in ["/content/akan-vits-finetuned/loss_curve.png",
          "/content/akan-vits-finetuned/losses.csv",
          "/content/akan-vits-finetuned/convergence_summary.json",
          "/content/train_akan.log"]:
    if os.path.exists(f):
        try:
            upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f),
                        repo_id="Dickson32-cell/akan-twi-mms", repo_type="model")
            print("uploaded:", os.path.basename(f))
        except Exception as e:
            print("upload failed:", os.path.basename(f), str(e)[:100])
print("\n\u2705 DONE - https://huggingface.co/Dickson32-cell/akan-twi-mms")
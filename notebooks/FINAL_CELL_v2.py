# Akan TTS - FINAL CELL v3 (heal-on-contact + drift-guard for ALL lazy VideoReader imports
# + truncating monitor + true-error labels + tfevents convergence + push)
# Two-line URL loader runs this. Self-contained; works in any kernel state.
import os, sys, json, glob, subprocess, time, urllib.request
from pathlib import Path

print("═══ AKAN TTS FINAL CELL v3 ═══", flush=True)

def stage(n, t): print("\n" + "="*14 + f"  STAGE {n}: {t}  " + "="*14, flush=True)

def shell(cmd, must=True, cwd=None, timeout=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd, timeout=timeout)
    if r.returncode != 0 and must:
        print("COMMAND FAILED:", cmd)
        print((r.stdout or "")[-1200:]); print((r.stderr or "")[-1200:])
        print(">>> paste the lines above to Hermes in Telegram, then re-run this cell.")
        raise SystemExit("STOPPED after failed command")
    return r

stage(0, "Kernel-state + file heal-check")
_vids = glob.glob("/usr/local/lib/python3*/dist-packages/datasets/features/video.py")
if _vids:
    _vid = _vids[0]
    _src = open(_vid, encoding="utf-8", errors="replace").read()
    def _ok(t, v=_vid):
        try: compile(t, v, "exec"); return True
        except Exception: return False
    if _ok(_src):
        print("video.py compiles OK (healed or healthy)")
    else:
        print("video.py wounded - healing TYPE_CHECKING span (v4)...")
        _lines = _src.splitlines(keepends=True)
        _ti = next(i for i, l in enumerate(_lines) if l.strip() == "if TYPE_CHECKING:")
        _fi = next(i for i in range(_ti, min(_ti + 14, len(_lines)))
                   if "from .features import FeatureType" in _lines[i])
        _canon = ["if TYPE_CHECKING:\n", "    try:\n",
                  "        from torchvision.io import VideoReader\n",
                  "    except ImportError:\n", "        VideoReader = None\n",
                  "\n", "    from .features import FeatureType\n"]
        _new = "".join(_lines[:_ti] + _canon + _lines[_fi + 1:])
        assert _ok(_new), "heal failed - paste output to Hermes"
        open(_vid, "w", encoding="utf-8").write(_new)
        print("HEALED + compile-verified")
_r = shell('python -c "import datasets; print(\'REIMPORT-OK\', datasets.__version__)"')
_rn = ((_r.stdout or "") + (_r.stderr or "")).strip().splitlines()
print("fresh-python probe:", (_rn[-1] if _rn else "(none)"))

stage(1, "GPU")
import torch
if not torch.cuda.is_available():
    raise SystemExit("NO GPU - Runtime menu > Change runtime type > T4 GPU, then re-run this cell")
print("GPU OK:", torch.cuda.get_device_name(0))

stage(2, "Versions (metadata) + harness present")
import importlib.metadata as md
_vt, _vd = md.version("transformers"), md.version("datasets")
print("pins:", _vt, "| datasets", _vd)
assert Path("/content/finetune-hf-vits/run_vits_finetuning.py").exists(), "harness missing - run the long loader once first"

stage(3, "HF login")
from huggingface_hub import HfApi
_a = HfApi()
try:
    print("logged in as:", _a.whoami()["name"])
except Exception:
    from huggingface_hub import notebook_login
    notebook_login()

stage(4, "Corpus + base present?")
assert Path("/content/tts_data").exists(), "tts_data missing - run the long loader once first"
assert glob.glob("/content/aka_train_base/*.safetensors"), "aka_train_base missing - run the long loader once first"
print("tts_data OK | base OK")

stage(6, "Config (gentle run)")
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
    print("checkpoints present -> RESUME")
else:
    cfg.pop("resume_from_checkpoint", None)
    print("no checkpoints -> FRESH RUN")
Path("/content/finetune_akan.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

stage(6.5, "Harness harness patches verified")
_H = "/content/finetune-hf-vits/run_vits_finetuning.py"
_hs = open(_H, encoding="utf-8").read()
print("load_from_disk patched:", "load_from_disk(data_args.dataset_name)" in _hs)
print("int-cast patched:", "int(max_tokens_length)" in _hs)
assert "load_from_disk(data_args.dataset_name)" in _hs, "harness patches missing - run the long loader first"

stage(6.7, "DRIFT-GUARD: patch ALL lazy VideoReader imports (dataset formatters)")
_fixed = False
_formatters = glob.glob("/usr/local/lib/python3*/dist-packages/datasets/formatting/*.py") \
            + glob.glob("/usr/lib/python3*/dist-packages/datasets/formatting/*.py")
for _fp in _formatters:
    _t = open(_fp, encoding="utf-8", errors="replace").read()
    if "from torchvision.io import VideoReader" in _t:
        _L = _t.splitlines(keepends=True)
        _out = []
        for _l in _L:
            if "from torchvision.io import VideoReader" in _l:
                _ind = _l[:len(_l) - len(_l.lstrip())]
                _out += [_ind + "try:", _ind + "    from torchvision.io import VideoReader",
                         _ind + "except ImportError:", _ind + "    VideoReader = None"]
            elif "isinstance(value, VideoReader)" in _l and "VideoReader is not None" not in _l:
                _out.append(_l.replace("isinstance(value, VideoReader)",
                                       "VideoReader is not None and isinstance(value, VideoReader)"))
            else:
                _out.append(_l)
        _new = "".join(_out)
        compile(_new, _fp, "exec")
        open(_fp, "w", encoding="utf-8").write(_new)
        print("drift-guard PATCHED:", os.path.basename(_fp))
        _fixed = True
if not _fixed:
    print("formatters already drift-free")
# TRUE reproduction probe: fresh python WITH torchvision loaded + tensorize (the exact crash path)
_pr = shell('python -c "import torchvision; import datasets.formatting.np_formatter as m; f=m.NumpyFormatter(); f._tensorize([1,2,3]); print(\'TENSORIZE-OK\')"', must=False)
_ln = ((_pr.stdout or "") + (_pr.stderr or "")).strip().splitlines()
print("formatter probe:", (_ln[-1][:110] if _ln else "(none)"))
if _pr.returncode != 0:
    print("\n".join(_ln[-12:]))
    raise SystemExit("formatter probe failed - paste lines above to Hermes")

stage(7, "Training (truncating monitor: only THIS attempt shows)")
LOG = "/content/train_akan.log"
_MARK = "@@@@@@ ATTEMPT-MARKER %d @@@@@@" % time.time()
with open(LOG, "a") as lf: lf.write("\n" + _MARK + "\n")
cmd = ["python", "-m", "accelerate.commands.launch",
       "/content/finetune-hf-vits/run_vits_finetuning.py",
       "/content/finetune_akan.json"]
print("launch args:", cmd)
proc = subprocess.Popen(cmd, stdout=open(LOG, "a"), stderr=subprocess.STDOUT,
                        bufsize=1, universal_newlines=True, cwd="/content/finetune-hf-vits")
Path("/content/train_pid.txt").write_text(str(proc.pid))
print("launched pid:", proc.pid, "- loss lines appear in a few minutes; runs for hours")

_lines_all, _seen, _last = Path(LOG).read_text(errors="replace").splitlines(), False, 0
while proc.poll() is None:
    time.sleep(30)
    _lines_all = Path(LOG).read_text(errors="replace").splitlines()
    if not _seen:
        _skip = next((i for i, l in enumerate(_lines_all) if l.startswith("@@@@@@ ATTEMPT-MARKER")), None)
        if _skip is None: continue
        _last, _seen = _skip + 1, True
    for l in _lines_all[_last:]:
        _ll = l.lower()
        if any(k in _ll for k in ("loss", "eval_", "epoch", "it/s", "error", "traceback")):
            print(l[:170], flush=True)
    _last = len(_lines_all)
rc = proc.returncode
print("\n=== TRAINING EXITED rc=%s ===" % rc, flush=True)
if rc != 0:
    _m = [i for i, l in enumerate(_lines_all) if l.startswith("@@@@@@ ATTEMPT-MARKER")]
    _seg = _lines_all[(_m[-1] + 1) if _m else max(0, len(_lines_all) - 40):]
    print("THIS ATTEMPT's LAST 25 LINES (the true error):")
    for l in _seg[-25:]: print("   ", l[:170])
    raise SystemExit("TRAINING FAILED - paste the THIS ATTEMPT block to Hermes")

stage(8, "Convergence report (tfevents + CSV)")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
_ev = sorted(glob.glob("/content/akan-vits-finetuned/**/events.out.tfevents.*", recursive=True))
if not _ev:
    print("no tfevents found - inspect training output")
else:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(os.path.dirname(_ev[-1])); acc.Reload()
    tags = acc.Tags().get("scalars", [])
    CURVES = ["train_summed_losses", "train_loss_mel", "train_loss_kl",
              "train_loss_gen", "train_loss_disc", "train_loss_duration", "train_loss_fmaps"]
    _use = [t for t in CURVES if t in tags] or tags[:5]
    print("tags:", _use)
    fig, ax = plt.subplots(figsize=(10, 5))
    for t in _use:
        pts = [(e.step, e.value) for e in acc.Scalars(t)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.2, label=t.replace("train_loss_", ""))
    try: ax.set_yscale("log")
    except Exception: pass
    ax.set_xlabel("step"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig("/content/akan-vits-finetuned/loss_curve.png", dpi=130)
    import csv
    _maps = [dict((e.step, e.value) for e in acc.Scalars(t)) for t in _use]
    with open("/content/akan-vits-finetuned/losses.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["step"] + _use)
        for s_ in sorted(_maps[0]):
            w.writerow([s_] + [m.get(s_) for m in _maps])
    vals = [e.value for e in acc.Scalars(_use[0])]
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

stage(9, "Save + push")
if os.path.exists("/content/drive/MyDrive"):
    os.makedirs("/content/drive/MyDrive/UG_TTS", exist_ok=True)
    subprocess.run("zip -qr '/content/drive/MyDrive/UG_TTS/akan_tts_ckpt_new.zip' /content/akan-vits-finetuned || true", shell=True)
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
print("Send the VERDICT line to Hermes.")

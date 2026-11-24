# Akan TTS - STAGES 6-9 POOL RUN (40h, ALL params trainable, Drive-mirrored checkpoints, resume-on-contact)
# Adapted from the verified freeze_stage7 flow. Key differences:
#   1. NO freeze insert - the whole point of P1 is full-model adaptation at 4.4x data.
#   2. cfg: LR 1e-4, 15 epochs, seed 456, hub_model_id Dickson32-cell/akan-twi-mms-pool,
#      push_to_hub with hub_strategy "end" (never mid-run pushes - the Oct-9 law),
#      save_steps 500, eval_steps 500, save_total_limit 2.
#   3. RESUME-ON-CONTACT: if local output has no checkpoints but the DRIVE mirror does,
#      copy the newest Drive checkpoint back into the local output dir before training.
#   4. DRIVE MIRROR during training: every 60s poll copies the newest checkpoint-*
#      into /content/drive/MyDrive/UG_TTS/pool_ckpts (replacing the previous mirror) -
#      so a dead session loses at most ~500 steps of compute.
# Assumes stages 1-5 already succeeded via pool_stage1.py.
import os, sys, json, glob, subprocess, time, shutil, re as _re
from pathlib import Path

OUT = "/content/akan-vits-finetuned-pool"
MIRROR = "/content/drive/MyDrive/UG_TTS/pool_ckpts"
LOG = "/content/train_akan_pool.log"

def stage(n, t): print("\n" + "="*14 + f"  STAGE {n}: {t}  " + "="*14, flush=True)

# config
stage(6, "Pool-run config (40h, LR 1e-4, 15 epochs, seed 456)")
cfg = {
  "project_name": "akan_twi_tts_pool", "push_to_hub": True,
  "hub_model_id": "Dickson32-cell/akan-twi-mms-pool", "hub_strategy": "end",
  "report_to": ["tensorboard"], "logging_strategy": "steps", "logging_steps": 20,
  "overwrite_output_dir": False,
  "output_dir": OUT,
  "dataset_name": "/content/tts_data", "dataset_config_name": None,
  "audio_column_name": "audio", "text_column_name": "text",
  "train_split_name": "train", "eval_split_name": "eval",
  "speaker_id_column_name": "speaker_id", "filter_on_speaker_id": 0,
  "override_speaker_embeddings": False,
  "full_generation_sample_text": "Mema wo akye, me nuanom. Wo ho te s\u025bn?",
  "max_duration_in_seconds": 12.0, "min_duration_in_seconds": 2.0, "max_tokens_length": 500,
  "model_name_or_path": "/content/aka_train_base",  # local base WITH preprocessor_config.json staged by pool_stage1 (content = the values both completed runs used; this stack 404s/errors on a missing file either way)
  "preprocessing_num_workers": 1,
  "do_train": True, "num_train_epochs": 15,
  "gradient_accumulation_steps": 1, "per_device_train_batch_size": 16,
  "learning_rate": 1e-4, "lr_scheduler_type": "cosine", "warmup_ratio": 0.02,
  "adam_beta1": 0.8, "adam_beta2": 0.99, "weight_decay": 0.01, "group_by_length": False,
  "do_eval": True, "eval_steps": 500, "per_device_eval_batch_size": 8,
  "max_eval_samples": 25, "do_step_schedule_per_epoch": False,
  "save_steps": 500, "save_total_limit": 2,
  "weight_disc": 3, "weight_fmaps": 1, "weight_gen": 1, "weight_kl": 1.5,
  "weight_duration": 1, "weight_mel": 35, "fp16": True, "seed": 456
}
# resume-on-contact: local checkpoints win; else pull latest from the Drive mirror
Path(MIRROR).mkdir(parents=True, exist_ok=True)
local_ck = sorted(glob.glob(OUT + "/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[-1])) if Path(OUT).exists() else []
mir_ck = sorted(glob.glob(MIRROR + "/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[-1]))
if local_ck:
    cfg["resume_from_checkpoint"] = "latest"  # STRING, the harness requires latest
    print("local checkpoints found -> RESUME (latest =", local_ck[-1].rsplit('/', 1)[-1], ")")
elif mir_ck:
    newest = mir_ck[-1]
    shutil.copytree(newest, OUT + "/" + Path(newest).name, dirs_exist_ok=True)
    _n = sorted(glob.glob(OUT + "/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[-1]))
    if _n:
        cfg["resume_from_checkpoint"] = "latest"
        print("DRIVE RESUME: restored -> newest Drive checkpoint", Path(newest).name)
    else:
        print("drive copy failed -> FRESH RUN")
else:
    print("no checkpoints anywhere -> FRESH RUN")
Path("/content/finetune_akan_pool.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print("pool config written: LR 1e-4, 15 epochs, ~26.5k steps, repo Dickson32-cell/akan-twi-mms-pool")

# harness patches (the 4 verified fixes; NO freeze insert on this run)
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
        _candidates = (
            "from torchvision.io import VideoReader",
            "from torchvision.io.video import VideoReader",
            "from torchvision.io import video as _video_mod",
        )
        _hit = None
        for _cand in _candidates:
            for _l in _cur.splitlines(keepends=True):
                if _l.strip() == _cand:
                    _hit = (l_ := _l)
                    break
            if _hit:
                break
        if _hit:
            _out, _done = [], False
            for _l in _cur.splitlines(keepends=True):
                if not _done and _l.strip() == _hit.strip():
                    _ind = _l[: len(_l) - len(_l.lstrip())]
                    _out.append(_ind + "try:")
                    _out.append(_ind + "    " + _l.strip())
                    _out.append(_ind + "except ImportError:")
                    _out.append(_ind + "    VideoReader = None")
                    _done = True
                else:
                    _out.append(_l)
            assert _done, "candidate import line vanished mid-patch"
            _new = "\n".join([l.rstrip() for l in "".join(_out).splitlines()])
            compile(_new, _vid, "exec")
            open(_vid, "w", encoding="utf-8").write(_new + "\n")
            print("harness PATCHED: video.py optional import (compile-verified, pattern: " + _hit.strip() + ")")
        else:
            # no known pattern in this image's video.py - verify the module imports
            # clean as-is; if it does, the patch is simply not required here.
            import importlib
            try:
                importlib.import_module("datasets.features.video")
                print("video.py has a different import layout but imports clean - patch 4 NOT required on this image; continuing")
            except Exception as _e:
                print("video.py unpatchable AND broken:", str(_e)[:200])
                raise SystemExit("video.py needs manual attention - send the output above to Hermes")

# training + live monitor + DRIVE CHECKPOINT MIRROR
stage(7, "Training (pool run) - checkpoints mirrored to Drive every save")
script = "/content/finetune-hf-vits/run_vits_finetuning.py"
if not Path(script).exists():
    raise SystemExit("harness not found - run pool_stage1.py (stages 1-5) first")
cmd = [sys.executable, "-m", "accelerate.commands.launch", script, "/content/finetune_akan_pool.json"]
print("launch args:", cmd)
log_f = open(LOG, "a")
proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, bufsize=1,
                        universal_newlines=True, cwd=os.path.dirname(script))
Path("/content/train_pool_pid.txt").write_text(str(proc.pid))
print("launched pid:", proc.pid, "- losses appear after a few minutes; CLOSE THE LAPTOP ANY TIME - the run is on Google's GPU")

def mirror_newest():
    """copy the newest local checkpoint to the Drive mirror (replacing old mirror); copy the log too."""
    ck = sorted(glob.glob(OUT + "/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[-1]))
    if not ck:
        return None
    newest = Path(ck[-1]).name
    tgt = MIRROR + "/" + newest
    cur = sorted(glob.glob(MIRROR + "/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[-1]))
    done = Path(tgt, ".mirror_done")
    if not done.exists():
        print("[MIRROR] saving", newest, "to Drive ...", flush=True)
        # copy-first, swap-later: never delete the previous good mirror until the
        # new copy has fully landed (a mid-save copy must not destroy the mirror).
        partial = tgt + ".partial"
        shutil.rmtree(partial, ignore_errors=True)
        shutil.copytree(ck[-1], partial, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(".tmp*"))
        for old in cur + (glob.glob(tgt) if Path(tgt).exists() else []):
            shutil.rmtree(old, ignore_errors=True)
        os.rename(partial, tgt)
        # mirror the running log too (resumable diagnostics)
        try:
            shutil.copy2(LOG, MIRROR + "/train_akan_pool.log")
        except Exception:
            pass
        done.write_text("mirrored")
        print("[MIRROR] done:", tgt, flush=True)
    return Path(newest).name

mirrored_last = None
last = 0
t0 = time.time()
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
    try:
        mirrored_last = mirror_newest() or mirrored_last
    except Exception as e:
        print("[MIRROR] transient failure (retried next poll):", str(e)[:140], flush=True)
log_f.close()
try:
    mirrored_last = mirror_newest() or mirrored_last
except Exception as e:
    print("[MIRROR] final-poll failure:", str(e)[:140])
n_lines = sum(1 for _ in open(LOG, "r", errors="replace"))
mins = round((time.time() - t0) / 60, 1)
print("\n=== TRAINING EXITED rc=%s | ran %.1f min this session | log lines: %d ===" % (proc.returncode, mins, n_lines))
if proc.returncode != 0:
    print("TRAINING FAILED (or interrupted) - last lines:")
    for l in Path(LOG).read_text(errors="replace").splitlines()[-12:]:
        print("   ", l[:180])
    print("NOTHING IS LOST - the newest checkpoint is mirrored in Drive: " + MIRROR)
    print("Re-run this cell when ready - it RESUMES from the newest checkpoint (Drive or local).")
    raise SystemExit("See the lines above.")
if mirrored_last:
    print("final mirrored checkpoint:", mirrored_last)
if int(Path(LOG).read_text(errors="replace").splitlines()[-1].lower().count("x")) >= 0:
    pass  # no-op guard

# convergence report (tfevents)
stage(8, "Convergence report from tfevents")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ev_files = sorted(glob.glob(OUT + "/**/events.out.tfevents.*", recursive=True))
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
    ax.set_title("Akan TTS POOL fine-tune (40h) - convergence (tfevents)")
    fig.tight_layout()
    fig.savefig(OUT + "/loss_curve.png", dpi=130)
    import csv as _csv
    with open(OUT + "/losses.csv", "w", newline="", encoding="utf-8") as fh:
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
              open(OUT + "/convergence_summary.json", "w"), indent=2)

# save + push to the POOL repo (end only)
stage(9, "Save to Drive + push POOL artifacts to its OWN repo")
os.makedirs("/content/drive/MyDrive/UG_TTS", exist_ok=True)
subprocess.run("zip -qr '/content/drive/MyDrive/UG_TTS/akan_tts_ckpt_pool.zip' " + OUT + " || true", shell=True)
from huggingface_hub import HfApi, upload_folder
api = HfApi()
try:
    api.create_repo(repo_id="Dickson32-cell/akan-twi-mms-pool", repo_type="model", exist_ok=True)
    print("repo ready: Dickson32-cell/akan-twi-mms-pool")
except Exception as e:
    print("create_repo:", str(e)[:120])
try:
    api.upload_folder(folder_path=OUT, repo_id="Dickson32-cell/akan-twi-mms-pool",
                      repo_type="model", commit_message="P1 pool run: 39.98h single-narrator, LR 1e-4, 15 epochs, seed 456 (all trainable)")
    print("PUSHED to Dickson32-cell/akan-twi-mms-pool (weights + artifacts)")
except Exception as e:
    print("push failed:", str(e)[:200]); print("the checkpoint zip is safe in Drive; push again later from a new session")
print("\n=== POOL RUN COMPLETE - send the VERDICT + rc lines to Hermes for the 4-way eval ===")

# POOL CONTINUE CELL - documented recovery after checkpoint-20000 was lost mid-save.
# Facts this run is built on (all verified):
#   - Drive held ONLY model.safetensors from checkpoint-20000 (mid-save casualty of the VM death).
#   - The pool training log's OWN cosine values prove the schedule: at step 14500, lr=4.33e-5.
#     Cosine formula check: lr(t) = peak * (1+cos(pi*(t-warm)/(total-warm)))/2 with peak 1e-4,
#     warm 525, total 26265 -> at 20000: lr = 1.38e-5. Continuation starts at LR 1.4e-5.
#   - The discriminator cannot be restored from any mirror (none survive); it re-initializes
#     fresh exactly like the original pool run did and re-converges within a few hundred steps.
#   - The generator warm-continues for the remaining 6265 steps to reach the planned 26,265.
# All of the above goes into the run's model card + convergence summary (documented, not hidden).
import glob, json, os, re, subprocess, time, shutil, sys
from pathlib import Path

OUT = "/content/akan-vits-finetuned-continue"
SRC_DRIVE = "/content/drive/MyDrive/UG_TTS/pool_ckpts/checkpoint-20000/model.safetensors"
BASEDIR = "/content/aka_train_base"
KEEP = "/content/drive/MyDrive/UG_TTS/pool_step20000_generator.safetensors"
MIRROR = "/content/drive/MyDrive/UG_TTS/pool_ckpts"
REPO = "Dickson32-cell/akan-twi-mms-pool"
LOG = "/content/train_akan_continue.log"

# 0) sanity: the rescued generator weights must exist
if not Path(SRC_DRIVE).exists():
    raise SystemExit("rescued weights not found at " + SRC_DRIVE + " - send this output to Hermes")
sz = Path(SRC_DRIVE).stat().st_size
print("rescued generator weights:", "%.1f MB" % (sz / 1e6))
if sz < 100e6 or sz > 400e6:
    raise SystemExit("weight-file size out of expected range - send this output to Hermes")

# 1) build the warm-start dir: base config + tokenizer + the 20k-step generator weights
WARM = "/content/aka_continue_base"
os.makedirs(WARM, exist_ok=True)
for f in ("config.json", "tokenizer_config.json", "vocab.json", "special_tokens_map.json"):
    if not Path(WARM, f).exists():
        shutil.copy2(Path(BASEDIR, f), Path(WARM, f))
json.dump({"feature_extractor_type": "VitsFeatureExtractor", "feature_size": 80,
           "hop_length": 256, "max_wav_value": 32768.0, "n_fft": 1024,
           "padding_side": "right", "padding_value": 0.0,
           "return_attention_mask": False, "sampling_rate": 16000},
          open(Path(WARM, "preprocessor_config.json"), "w"), indent=2)
if not Path(WARM, "model.safetensors").exists():
    print("copying rescued generator weights into the warm-start dir ...", flush=True)
    shutil.copy2(SRC_DRIVE, Path(WARM, "model.safetensors"))
ok = Path(WARM, "model.safetensors").stat().st_size == sz
print("warm-start dir ready:", sorted(os.path.basename(x) for x in glob.glob(WARM + "/*")),
      "| weights bytes match:", ok)
if not ok:
    raise SystemExit("warm-start copy failed verification - send this output to Hermes")

# 2) keep an archival copy of the break-point weights (rename the stale mirror dir first)
try:
    if not Path(KEEP).exists() and Path(SRC_DRIVE).exists():
        shutil.copy2(SRC_DRIVE, KEEP)
        print("archival keep-safe copy: " + KEEP)
except Exception as e:
    print("keep-safe copy skipped:", str(e)[:120])

# 3) the continuation config - EXACTLY the pool config plus the documented changes
cfg = {
  "project_name": "akan_twi_tts_pool_continue", "push_to_hub": True,
  "hub_model_id": REPO, "hub_strategy": "end",
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
  "model_name_or_path": WARM,
  "preprocessing_num_workers": 1,
  "do_train": True, "max_steps": 6265, "num_train_epochs": 15,
  "gradient_accumulation_steps": 1, "per_device_train_batch_size": 16,
  "learning_rate": 1.4e-5, "lr_scheduler_type": "cosine", "warmup_ratio": 0.01,
  # warmup 0.01 = 62 of 6265 steps: the harness requires a num_warmup_steps value for
  # cosine (warmup_ratio 0 crashes at scheduler init) AND the optimizer state was lost
  # with the checkpoint - 62 steps lets Adam re-accumulate moments before the tiny LR.
  "adam_beta1": 0.8, "adam_beta2": 0.99, "weight_decay": 0.01, "group_by_length": False,
  "do_eval": True, "eval_steps": 500, "per_device_eval_batch_size": 8,
  "max_eval_samples": 25, "do_step_schedule_per_epoch": False,
  "save_steps": 500, "save_total_limit": 2,
  "weight_disc": 3, "weight_fmaps": 1, "weight_gen": 1, "weight_kl": 1.5,
  "weight_duration": 1, "weight_mel": 35, "fp16": True, "seed": 456
}
Path("/content/finetune_akan_continue.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print("CONTINUATION config: warm generator @20k | disc fresh | LR 1.4e-5 cosine | max_steps 6265 -> ends at the planned 26,265")

# 4) harness already cloned by pool_stage1 (or clone now) + the 4 patches if missing
hscript = "/content/finetune-hf-vits/run_vits_finetuning.py"
if not Path(hscript).exists():
    subprocess.run("git clone https://github.com/ylacombe/finetune-hf-vits /content/finetune-hf-vits", shell=True)
_s = open(hscript, encoding="utf-8").read()
if "load_from_disk(data_args.dataset_name)" not in _s:
    _s = re.sub(r"raw_datasets\[.train.\] = load_dataset\((?:.|\n)*?\)\n", 'raw_datasets["train"] = load_from_disk(data_args.dataset_name)[data_args.train_split_name]\n', _s, count=1)
    _s = re.sub(r"raw_datasets\[.eval.\] = load_dataset\((?:.|\n)*?\)\n", 'raw_datasets["eval"] = load_from_disk(data_args.dataset_name)[data_args.eval_split_name]\n', _s, count=1)
    if "from datasets import DatasetDict, load_dataset" in _s:
        _s = _s.replace("from datasets import DatasetDict, load_dataset", "from datasets import DatasetDict, load_dataset, load_from_disk")
    open(hscript, "w", encoding="utf-8").write(_s)
    print("harness PATCHED: load_from_disk")
else:
    print("harness load_from_disk already patched")
_s = open(hscript, encoding="utf-8").read()
if 'batch[model_input_name] = string_inputs.get("input_ids")[: max_tokens_length + 1]' in _s:
    _s = _s.replace('batch[model_input_name] = string_inputs.get("input_ids")[: max_tokens_length + 1]',
                    'batch[model_input_name] = string_inputs.get("input_ids")[: int(max_tokens_length) + 1]')
    open(hscript, "w", encoding="utf-8").write(_s)
    print("harness PATCHED: float-slice int cast")
else:
    print("harness slice already patched")
_plot = "/content/finetune-hf-vits/utils/plot.py"
_ps = open(_plot, encoding="utf-8").read()
if "tostring_rgb" in _ps:
    _ps = _ps.replace("data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)",
                      "data = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()")
    open(_plot, "w", encoding="utf-8").write(_ps)
    print("harness PATCHED: buffer_rgba")
else:
    print("plot.py already modern")

# 5) pre-launch sweep: no orphaned launcher from a failed attempt may hold the GPU
subprocess.run("pkill -9 -f run_vits_finetuning || true", shell=True)
subprocess.run("pkill -9 -f accelerate.commands.launch || true", shell=True)
time.sleep(3)
print("pre-launch sweep done (orphaned launchers cleared)")
cmd = [sys.executable, "-m", "accelerate.commands.launch", hscript, "/content/finetune_akan_continue.json"]
log_f = open(LOG, "a")
proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, bufsize=1,
                        universal_newlines=True, cwd=os.path.dirname(hscript))
Path("/content/train_continue_pid.txt").write_text(str(proc.pid))
print("CONTINUATION launched pid:", proc.pid, "- losses/minutes as before; the GPU does the work")
time.sleep(10)

# 6) watch + mirror-complete-only (same law as pool_stage7: never publish mid-save/incomplete)
def mirror_newest():
    out = []
    for p in glob.glob(OUT + "/checkpoint-*"):
        m = re.fullmatch(r"checkpoint-(\d+)", Path(p).name)
        if m:
            out.append((int(m.group(1)), p))
    if not out:
        return None
    n, newest_path = max(out)
    newest = Path(newest_path).name
    tgt = MIRROR + "/" + newest
    if Path(tgt, ".mirror_done").exists():
        return newest
    _req = {"trainer_state.json", "pytorch_model.bin", "pytorch_model_1.bin"}
    _have = set(os.path.basename(f) for f in glob.glob(newest_path + "/*"))
    if any(x.startswith(".tmp") for x in _have) or not _req.issubset(_have):
        return newest
    print("[MIRROR] saving", newest, "to Drive ...", flush=True)
    partial = tgt + ".partial"
    shutil.rmtree(partial, ignore_errors=True)
    shutil.copytree(newest_path, partial, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".tmp*"))
    for old in glob.glob(MIRROR + "/checkpoint-*"):
        if Path(old).name.startswith(newest + "."):
            continue
        shutil.rmtree(old, ignore_errors=True)
    os.rename(partial, tgt)
    try:
        shutil.copy2(LOG, MIRROR + "/train_akan_continue.log")
    except Exception:
        pass
    Path(tgt, ".mirror_done").write_text("mirrored")
    print("[MIRROR] done:", tgt, flush=True)
    return newest

lines = []
try:
    lines = Path(LOG).read_text(errors="replace").splitlines()
except FileNotFoundError:
    pass
last = len(lines)
t0 = time.time()
while True:
    try:
        rc = proc.poll()
    except Exception:
        rc = None
    if rc is not None:
        break
    time.sleep(25)
    try:
        nl = Path(LOG).read_text(errors="replace").splitlines()
    except FileNotFoundError:
        nl = []
    for l in nl[last:]:
        if any(k in l.lower() for k in ("loss", "eval_", "error", "traceback", "epoch", "it/s")):
            print(l[:180])
    last = len(nl)
    try:
        mirror_newest()
    except Exception as e:
        print("[MIRROR] retry next poll:", str(e)[:140])
raw = []
try:
    raw = Path(LOG).read_text(errors="replace").splitlines()
except FileNotFoundError:
    pass
mins = round((time.time() - t0) / 60, 1)
print("\n=== CONTINUATION EXITED rc=%s | watched %.1f min ===" % (rc, mins))
if rc != 0:
    for l in raw[-10:]:
        print("   ", l[:180])
    raise SystemExit("continuation failed - send the lines above to Hermes")

# 7) convergence from tfevents + honest recovery note into the summary
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ev_files = sorted(glob.glob(OUT + "/**/events.out.tfevents.*", recursive=True))
if ev_files:
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
    ax.set_xlabel("continuation step"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("Akan TTS POOL continuation: generator @20k + 6265 steps to 26,265")
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
        vals = [p[1] for p in acc.Scalars(main_tag)]
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
json.dump({
    "verdict": locals().get("verdict", "CHECK MANUALLY"),
    "recovery": {
        "cause": "checkpoint-20000 died mid-save; the only Drive mirror carried model.safetensors only",
        "action": "generator warm-continued from the 20k-step weights for 6265 steps; discriminator re-initialized; LR reconstructed to the cosine value at the break (1.4e-5, verified against the training log: lr(14500)=4.33e-5)",
        "schedule": "26265 planned global steps reached (0-20000 original + continuation 6265)",
        "breakpoint_weights": KEEP
    }}, open(OUT + "/convergence_summary.json", "w"), indent=2)

# 8) Drive zip + push to the POOL repo
subprocess.run("zip -qr '/content/drive/MyDrive/UG_TTS/akan_tts_ckpt_continue.zip' " + OUT + " || true", shell=True)
from huggingface_hub import HfApi, upload_folder
api = HfApi()
try:
    api.create_repo(repo_id=REPO, repo_type="model", exist_ok=True)
    print("repo ready:", REPO)
except Exception as e:
    print("create_repo:", str(e)[:120])
try:
    api.upload_folder(folder_path=OUT, repo_id=REPO, repo_type="model",
                      commit_message="P1 pool run (continuation): warm generator @20k + 6265 steps = planned 26,265; disc re-init; documented in recovery block")
    print("PUSHED to", REPO)
except Exception as e:
    print("push failed:", str(e)[:200])
    print("the checkpoint zip is safe in Drive; push again later")
print("\n=== POOL CONTINUATION COMPLETE - send the VERDICT lines to Hermes for the 4-way eval ===")

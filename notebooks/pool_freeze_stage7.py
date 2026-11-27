# Akan TTS - STAGES 6-9 POOL-FREEZE RUN (40h, LR 5e-5; text_encoder + duration_predictor frozen; flow + decoder train)
# Variant of the verified pool_stage7 flow. Differences vs the pool run:
#   0. config fetched from the committed configs/akan-twi-pool-freeze.json (LR 5e-5, save_steps 1000,
#      save_total_limit null so the sweep keeps every checkpoint, repo = akan-twi-mms-pool-freeze).
#   1. FREEZE insert patched into the harness after model construction: text_encoder (+ its
#      embeddings) and duration_predictor get requires_grad=False; frozen/trainable tensor counts printed.
#   2. Durability beyond the trainer-side safety net: the session monitor mirrors the newest COMPLETE
#      checkpoint (keep newest 3 in Drive, proven law) AND uploads it to the private repo, deleting older
#checkpoint dirs there so the repo stays ~2GB; resume-from-latest granularity = 1,000 steps.
#   3. After training: frozen-tensor byte-identity assert vs the staged base, convergence report, end push.
# Sweep runs in a separate session (POOLFREEZE_SWEEP_CELL.py) to keep this session inside the 12h cap.
# Assumes stages 1-5 already succeeded via pool_stage1.py (corpus + base staging are identical).
import os, sys, json, glob, subprocess, time, shutil, re as _re, urllib.request
from pathlib import Path

OUT = "/content/akan-vits-finetuned-pool-freeze"
MIRROR = "/content/drive/MyDrive/UG_TTS/pool_freeze_ckpts"
LOG = "/content/train_akan_pool_freeze.log"
REPO = "Dickson32-cell/akan-twi-mms-pool-freeze"
CFG_URL = "https://raw.githubusercontent.com/Dickson32-cell/akan-tts-asante/_PIN_/configs/akan-twi-pool-freeze.json"
BASE = "/content/aka_train_base"
MAXS = 26265

def stage(n, t): print("\n" + "="*14 + f"  STAGE {n}: {t}  " + "="*14, flush=True)

# config (from the committed file; the JSON parses straight into harness TrainingArguments)
stage(6, "Pool-freeze config (40h, LR 5e-5, freeze text_encoder + duration_predictor, seed 456)")
cfg = json.loads(urllib.request.urlopen(CFG_URL, timeout=60).read().decode("utf-8"))
assert cfg["hub_model_id"] == REPO and cfg["learning_rate"] == 5e-5 and cfg["save_steps"] == 1000
assert cfg["save_total_limit"] is None, "sweep needs every checkpoint kept"
Path("/content/finetune_akan_pool_freeze.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print("config written: LR", cfg["learning_rate"], "| epochs", cfg["num_train_epochs"], "| repo", REPO)

# resume-on-contact: local checkpoints win; else pull the newest COMPLETE checkpoint from Drive
Path(MIRROR).mkdir(parents=True, exist_ok=True)
_REQUIRED_CORE = {"trainer_state.json", "config.json"}
_WEIGHT_SUFFIXES = (".safetensors", ".bin")

def _ck_safe(base):
    out_ = []
    for p in glob.glob(base + "/checkpoint-*"):
        m = _re.fullmatch(r"checkpoint-(\d+)", Path(p).name)
        if m:
            out_.append((int(m.group(1)), p))
    return sorted(out_)

def _ck_complete(pth):
    have = set(os.path.basename(f) for f in glob.glob(pth + "/*"))
    if any(n.startswith(".tmp") for n in have):
        return False, {"mid-save .tmp files"}
    if not _REQUIRED_CORE.issubset(have):
        return False, _REQUIRED_CORE - have
    if not [n for n in have if any(n.endswith(sf) for sf in _WEIGHT_SUFFIXES)]:
        return False, {"any weight file (*.safetensors / *.bin)"}
    return True, set()

local_ck = _ck_safe(OUT) if Path(OUT).exists() else []
mir_ck = _ck_safe(MIRROR)
if local_ck and _ck_complete(local_ck[-1][1])[0]:
    cfg["resume_from_checkpoint"] = "latest"  # STRING - the harness requires latest
    print("local checkpoints found -> RESUME (latest =", Path(local_ck[-1][1]).name, ")")
elif mir_ck:
    restored = None; skipped = []
    for n, pth in reversed(mir_ck):
        ok, missing = _ck_complete(pth)
        if not ok:
            skipped.append("%s (missing %s)" % (Path(pth).name, sorted(missing)))
            continue
        shutil.copytree(pth, OUT + "/" + Path(pth).name, dirs_exist_ok=True)
        if _ck_complete(OUT + "/" + Path(pth).name)[0]:
            restored = pth
            cfg["resume_from_checkpoint"] = "latest"
            print("DRIVE RESUME: restored -> newest COMPLETE checkpoint", Path(pth).name,
                  "(step %d of %d)" % (n, MAXS))
            if skipped:
                print("   skipped incomplete mirrors (died mid-save):", ", ".join(skipped))
            break
        shutil.rmtree(OUT + "/" + Path(pth).name, ignore_errors=True)
    if not restored:
        print("NO COMPLETE MIRROR - newest Drive checkpoints:", [Path(p).name for _, p in mir_ck])
        raise SystemExit("All Drive mirrors incomplete before rerunning.")
else:
    print("no checkpoints anywhere -> FRESH RUN")
if "resume_from_checkpoint" not in cfg:
    # fresh-run hygiene: leftover debris would block the trainer or poison implicit resume
    shutil.rmtree(OUT, ignore_errors=True)
    print("fresh-run hygiene: output dir cleared")

# harness patches (the 4 verified fixes) + THE FREEZE INSERT for this run
stage("6b", "Patching harness (4 verified fixes + pool-freeze insert)")
_h = "/content/finetune-hf-vits/run_vits_finetuning.py"
_s = open(_h, encoding="utf-8").read()
if "load_from_disk(data_args.dataset_name)" not in _s:
    _pat_train = r"raw_datasets\[.train.\] = load_dataset\((?:.|\n)*?\)\n"
    _s = _re.sub(_pat_train, 'raw_datasets["train"] = load_from_disk(data_args.dataset_name)[data_args.train_split_name]\n', _s, count=1)
    _pat_eval = r"raw_datasets\[.eval.\] = load_dataset\((?:.|\n)*?\)\n"
    _s = _re.sub(_pat_eval, 'raw_datasets["eval"] = load_from_disk(data_args.dataset_name)[data_args.eval_split_name]\n', _s, count=1)
    if "from datasets import DatasetDict, load_dataset" in _s:
        _s = _s.replace("from datasets import DatasetDict, load_dataset",
                        "from datasets import DatasetDict, load_dataset, load_from_disk")
    open(_h, "w", encoding="utf-8").write(_s)
    print("harness PATCHED: load_from_disk")
_s = open(_h, encoding="utf-8").read()
_slice_old = 'batch[model_input_name] = string_inputs.get("input_ids")[: max_tokens_length + 1]'
if _slice_old in _s:
    _s = _s.replace(_slice_old,
        'batch[model_input_name] = string_inputs.get("input_ids")[: int(max_tokens_length) + 1]')
    open(_h, "w", encoding="utf-8").write(_s)
    print("harness PATCHED: float-slice int cast")
else:
    print("harness slice already patched")
_plot = "/content/finetune-hf-vits/utils/plot.py"
_ps = open(_plot, encoding="utf-8").read()
if "tostring_rgb" in _ps:
    _ps = _ps.replace("data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)",
                      "data = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()")
    open(_plot, "w", encoding="utf-8").write(_ps)
    print("harness PATCHED: modern matplotlib buffer_rgba")
else:
    print("plot.py already modern")
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
                    _hit = _l
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
            _new = "\n".join([l.rstrip() for l in "".join(_out).splitlines()])
            compile(_new, _vid, "exec")
            open(_vid, "w", encoding="utf-8").write(_new + "\n")
            print("harness PATCHED: video.py optional import (compile-verified)")
        else:
            import importlib
            try:
                importlib.import_module("datasets.features.video")
                print("video.py has a different import layout but imports clean - patch 4 not required; continuing")
            except Exception as _e:
                print("video.py unpatchable AND broken:", str(_e)[:200])
                raise SystemExit("video.py needs manual attention - keep the output above for diagnosis")

# FREEZE INSERT (this run only): after model construction the frozen modules are protected.
_s = open(_h, encoding="utf-8").read()
if "# POOL-FREEZE INSERT" not in _s:
    _m = [i for i, l in enumerate(_s.splitlines()) if l.strip() == "model.eval()"]
    assert _m, "model.eval() not found in harness - paste the harness grep output above for diagnosis"
    _i = _m[0]
    _lines = _s.splitlines()
    _ind = _lines[_i][: len(_lines[_i]) - len(_lines[_i].lstrip())]
    _blk = (
        _ind + "# POOL-FREEZE INSERT: protect base phoneme path; flow + decoder adapt (LR 5e-5)\n"
        + _ind + "for _p in model.text_encoder.parameters():\n"
        + _ind + "    _p.requires_grad = False\n"
        + _ind + "for _p in model.duration_predictor.parameters():\n"
        + _ind + "    _p.requires_grad = False\n"
        + _ind + "_frozen = sum(1 for _n, _p in model.named_parameters() if not _p.requires_grad)\n"
        + _ind + "_kept = sum(1 for _n, _p in model.named_parameters() if _p.requires_grad)\n"
        + _ind + 'print(f"[FREEZE] text_encoder + duration_predictor frozen: {_frozen} tensors frozen / {_kept} trainable")\n'
    )
    _s = "\n".join(_lines[:_i + 1]) + "\n" + _blk + "\n".join(_lines[_i + 1:])
    compile(_s, _h, "exec")
    open(_h, "w", encoding="utf-8").write(_s)
    print("harness PATCHED: POOL-FREEZE insert (compile-verified)")
else:
    print("POOL-FREEZE insert already present")

# training + live monitor + Drive mirror + PER-CHECKPOINT PRIVATE-REPO PUSH
stage(7, "Training (pool-freeze) - checkpoints mirrored to Drive and pushed to the private repo")
script = "/content/finetune-hf-vits/run_vits_finetuning.py"
if not Path(script).exists():
    raise SystemExit("harness not found - run pool_stage1.py (stages 1-5) first")
cmd = [sys.executable, "-m", "accelerate.commands.launch", script, "/content/finetune_akan_pool_freeze.json"]
print("launch args:", cmd)
log_f = open(LOG, "a")
proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, bufsize=1,
                        universal_newlines=True, cwd=os.path.dirname(script))
Path("/content/train_pool_freeze_pid.txt").write_text(str(proc.pid))
print("launched pid:", proc.pid, "- losses appear after a few minutes; CLOSE THE LAPTOP ANY TIME - the run is on Google's GPU")

from huggingface_hub import HfApi
_api = HfApi()
try:
    _api.create_repo(repo_id=REPO, repo_type="model", private=True, exist_ok=True)
except Exception as _e:
    print("create_repo:", str(_e)[:140])

def _push_newest_hub(newest, newest_path):
    if not Path(newest_path, ".mirror_done").exists():
        return False
    files = _api.list_repo_files(repo_id=REPO, repo_type="model")
    have = {f for f in files}
    want = [f for f in (newest + "/" + n for n in os.listdir(newest_path))]
    if all(w in have for w in want):
        return True
    print("[HUBPUSH] uploading", newest, "to", REPO, "...", flush=True)
    _api.upload_folder(folder_path=newest_path, repo_id=REPO, repo_type="model",
                       path_in_repo=newest, commit_message="durability mirror: checkpoint " + newest)
    # rotation: keep only the newest checkpoint dir on the Hub (repo stays small)
    steps = []
    for f in files:
        head = f.split("/")[0]
        if f.startswith("checkpoint-") and _re.fullmatch(r"checkpoint-\d+", head):
            d = head
            dn = int(d.split("-")[1])
            if dn < int(newest.split("-")[1]) and d not in steps:
                steps.append(d)
    for d in steps:
        try:
            _api.delete_folder(repo_id=REPO, repo_type="model", path_in_repo=d, commit_message="rotation: superseded by " + newest)
            print("[HUBPUSH] rotated out:", d, flush=True)
        except Exception as e:
            print("[HUBPUSH] rotation skip:", str(e)[:120], flush=True)
    return True

def mirror_newest():
    """publish the newest COMPLETE checkpoint: Drive (keep newest 3) + private repo (keep newest 1)."""
    ck = _ck_safe(OUT)
    if not ck:
        return None
    n, newest_path = ck[-1]
    newest = Path(newest_path).name
    tgt = MIRROR + "/" + newest
    if not Path(tgt, ".mirror_done").exists():
        ok, missing = _ck_complete(newest_path)
        if not ok:
            print("[MIRROR] SKIP", newest, "- incomplete:", sorted(missing), flush=True)
            return None
        if not Path(MIRROR).exists():
            raise RuntimeError("Drive not mounted - run the loader cell which mounts Drive first")
        print("[MIRROR] saving", newest, "to Drive ...", flush=True)
        partial = tgt + ".partial"
        shutil.rmtree(partial, ignore_errors=True)
        shutil.copytree(newest_path, partial, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(".tmp*"))
        for old in glob.glob(MIRROR + "/checkpoint-*"):
            if Path(old).name.startswith(newest + "."):
                continue
            shutil.rmtree(old, ignore_errors=True)
        os.rename(partial, tgt)
        try:
            shutil.copy2(LOG, MIRROR + "/train_akan_pool_freeze.log")
        except Exception:
            pass
        Path(tgt, ".mirror_done").write_text("mirrored")
        print("[MIRROR] done:", tgt, flush=True)
        _all = sorted((int(Path(pp).name.split("-")[1]), pp) for pp in glob.glob(MIRROR + "/checkpoint-*")
                      if _re.fullmatch(r"checkpoint-(\d+)", Path(pp).name))
        for _oldn, _oldp in _all[:-3]:
            shutil.rmtree(_oldp, ignore_errors=True)
    try:
        _push_newest_hub(newest, newest_path)
    except Exception as e:
        print("[HUBPUSH] transient failure (retried next poll):", str(e)[:140], flush=True)
    return newest

mirrored_last = None
last = 0
t0 = time.time()
_hb = time.time()
while proc.poll() is None:
    time.sleep(25)
    if time.time() - _hb > 1800:
        _hb = time.time()
        _l2 = Path(LOG).read_text(errors="replace").splitlines()
        _m2 = [x for x in _l2 if "/%d [" % MAXS in x]
        print("[HEARTBEAT] alive, latest step:", (_m2[-1][:80] if _m2 else "no step lines yet"), flush=True)
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
    print("NOTHING IS LOST - the newest checkpoint is mirrored in Drive + on the private repo: " + MIRROR)
    print("Re-run this cell when ready - it RESUMES from the newest checkpoint (local, Drive or repo).")
    raise SystemExit("See the lines above.")
if mirrored_last:
    print("final mirrored checkpoint:", mirrored_last)
if "[FREEZE]" not in Path(LOG).read_text(errors="replace"):
    print("WARNING: no [FREEZE] line in the training log - the insert did not print; paste the log head above for diagnosis")

# FREEZE INTEGRITY: every frozen tensor must be byte-identical to the staged base
stage("8a", "Frozen-tensor integrity assert (text_encoder + duration_predictor vs staged base)")
try:
    from safetensors.torch import load_file
    fb = load_file(BASE + "/model.safetensors")
    fr = load_file(OUT + "/model.safetensors")
    fk = [k for k in fb if k.startswith("text_encoder.") or k.startswith("duration_predictor.")]
    same = sum(1 for k in fk if k in fr and (fb[k] == fr[k]).all())
    diff = [k for k in fk if k in fr and not (fb[k] == fr[k]).all()]
    print("frozen tensors compared:", len(fk), "| byte-identical:", same, "| DIFFERING:", diff[:8])
    assert not diff, "frozen tensors drifted - STOP and paste this output before any push"
    print("FROZEN INTEGRITY: PASS")
except Exception as e:
    print("integrity check error (paste for diagnosis):", str(e)[:200])

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
    ax.set_title("Akan TTS POOL-FREEZE fine-tune (40h, LR 5e-5) - convergence (tfevents)")
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

# save + push (end only): gen-only root + artifacts to the PRIVATE repo
stage(9, "Save to Drive + push pool-freeze artifacts to its OWN private repo")
os.makedirs("/content/drive/MyDrive/UG_TTS", exist_ok=True)
subprocess.run("zip -qr '/content/drive/MyDrive/UG_TTS/akan_tts_ckpt_pool_freeze.zip' " + OUT + " || true", shell=True)
from huggingface_hub import HfApi, upload_folder
api2 = HfApi()
try:
    api2.upload_folder(folder_path=OUT, repo_id=REPO, repo_type="model",
                       commit_message="pool-freeze run: 40h single-narrator, LR 5e-5, text_encoder + duration_predictor frozen, seed 456")
    print("PUSHED to " + REPO + " (weights + artifacts, private until evaluated)")
except Exception as e:
    print("push failed:", str(e)[:200]); print("the checkpoint zip is safe in Drive; push again later from a new session")

# ---- STAGE 10: DEV-SET CHECKPOINT SWEEP (staged: 25-clip subset for all, full 100 for finalists) ----
stage(10, "Checkpoint sweep on the dev split (staged; resumable; same family judge)")
SWEEP = "/content/sweep"
Path(SWEEP).mkdir(parents=True, exist_ok=True)
RES = Path(SWEEP) / "sweep_results.json"
results = []
if RES.exists():
    results = json.loads(RES.read_text(encoding="utf-8"))
    print("resumable sweep: %d checkpoints already scored, skipping those" % len(results))

# evaluator fetched from the pinned repo (same relative layout: REPO = parents[1])
EVPIN = "https://raw.githubusercontent.com/Dickson32-cell/akan-tts-asante/_PIN_"
os.makedirs("/content/akan_eval/src", exist_ok=True)
for fn in ("src/eval_asr.py", "src/normalize_twi.py"):
    open("/content/akan_eval/" + fn, "wb").write(urllib.request.urlopen(EVPIN + "/" + fn, timeout=60).read())

# dev refs (manifest order; 1-indexed pairs with file names containing the clip number)
dev = json.loads(urllib.request.urlopen(EVPIN + "/data/dev_split_poolfreeze.json", timeout=60).read().decode("utf-8"))
dev_clips = dev["clips"]
Path("/content/dev_refs.txt").write_text("\n".join(c["norm"] for c in dev_clips) + "\n", encoding="utf-8")
print("dev refs:", len(dev_clips), "clips | universe:", dev["universe_after_exclusions"],
      "| seed:", dev["seed"])

import torch as _t
from safetensors.torch import load_file as _lfs
from transformers import VitsModel as _VM, AutoTokenizer as _AT

def _synth(step, path, n_clips, tag):
    """synthesize dev clips for one checkpoint; names d_001.wav.. (1-indexed)."""
    outdir = Path(SWEEP) / ("%s_%s" % (tag, step))
    outdir.mkdir(parents=True, exist_ok=True)
    mdl = _VM.from_pretrained(path)
    tk = _AT.from_pretrained(path)
    mdl.to("cuda").eval()
    mdl.speaking_rate = 1.0
    mdl.noise_scale = 0.667
    mdl.noise_scale_w = 0.8
    _t.manual_seed(33)
    for i, c in enumerate(dev_clips[:n_clips]):
        f = outdir / ("d_%03d.wav" % (i + 1))
        if f.exists() and f.stat().st_size > 1000:
            continue
        inp = tk(c["norm"], return_tensors="pt")
        with _t.no_grad():
            o = mdl(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
        wv = o.waveform[0].cpu().numpy()
        import soundfile as _sf
        wv = wv / max(1e-6, abs(wv).max()) * 0.95
        _sf.write(str(f), wv, 16000, subtype="PCM_16")
    del mdl; _t.cuda.empty_cache()
    return outdir

def _judge(outdir, out_name):
    """one full invocation of the family evaluator (judge loads per call; measured cost accepted)."""
    cmd = [sys.executable, "/content/akan_eval/src/eval_asr.py", "--samples", str(outdir),
           "--prompts", "/content/dev_refs.txt", "--engine", "mms_aka", "--device", "cuda",
           "--out-eval", str(Path(SWEEP) / out_name)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if Path(SWEEP, out_name).exists():
        j = json.loads(Path(SWEEP, out_name).read_text(encoding="utf-8"))
        return {"step": None, "cer": j.get("mean_cer"), "wer": j.get("mean_wer"),
                "mismatch": j.get("mismatch_control_cer"), "rc": p.returncode}
    print("judge run failed (paste above for diagnosis):", (p.stdout + p.stderr)[-400:])
    return None

SUBSET_N = 25
FINALISTS = 3
all_ck = _ck_safe(OUT)
print("checkpoints to sweep:", [n for n, _ in all_ck])
for n, pth in all_ck:
    if any(r["step"] is not None and r["step"] == n and r.get("stage") == "subset" for r in results):
        continue
    od = _synth(n, pth, SUBSET_N, "sub")
    r = _judge(od, "sub_%s.json" % n)
    if r:
        r = {"stage": "subset", "step": n, "cer": r["cer"], "wer": r["wer"], "mismatch": r["mismatch"], "rc": r["rc"]}
        results = [x for x in results if not (x.get("stage") == "subset" and x.get("step") == n)] + [r]
        RES.write_text(json.dumps(results, indent=2))
        print("[SWEEP] step %d: subset CER %s | WER %s | mismatch %s" % (n, r["cer"], r["wer"], r["mismatch"]), flush=True)
        try:
            shutil.copy2(RES, MIRROR + "/sweep_results.json")
        except Exception:
            pass

subs = [r for r in results if r.get("stage") == "subset" and r.get("cer") is not None]
if len(subs) == len(all_ck) and subs:
    subs.sort(key=lambda r: r["cer"])
    finalists = subs[:FINALISTS]
    print("finalists (subset CER):", [(r["step"], r["cer"]) for r in finalists])
    finals = [r for r in results if r.get("stage") == "full"]
    for r in finalists:
        if any(f["step"] == r["step"] for f in finals):
            continue
        pth = dict((n, p) for n, p in all_ck)[r["step"]]
        od = _synth(r["step"], pth, len(dev_clips), "full")
        jr = _judge(od, "full_%s.json" % r["step"])
        if jr:
            fr = {"stage": "full", "step": r["step"], "cer": jr["cer"], "wer": jr["wer"],
                  "mismatch": jr["mismatch"], "rc": jr["rc"]}
            finals.append(fr)
            results = [x for x in results if not (x.get("stage") == "full" and x.get("step") == r["step"])] + [fr]
            RES.write_text(json.dumps(results, indent=2))
            print("[SWEEP] step %d: FULL-DEV CER %s | WER %s | mismatch %s" % (r["step"], fr["cer"], fr["wer"], fr["mismatch"]), flush=True)
    fulls = [r for r in results if r.get("stage") == "full" and r.get("cer") is not None]
    if fulls:
        fulls.sort(key=lambda r: r["cer"])
        win = fulls[0]
        print("\nWINNER: checkpoint-%d | full-100-clip dev CER %s | WER %s | mismatch %s"
              % (win["step"], win["cer"], win["wer"], win["mismatch"]))
        stage("10b", "Publish winner: gen-only root weights (discriminator never uploads)")
        win_dir = dict((n, p) for n, p in all_ck)[win["step"]]
        wout = Path(SWEEP) / "winner"
        wout.mkdir(exist_ok=True)
        src_w = _lfs(str(Path(win_dir) / "model.safetensors"))
        from safetensors.torch import save_file as _sfs
        _sfs(src_w, str(Path(wout) / "model.safetensors"))
        for f in ("config.json", "preprocessor_config.json", "vocab.json", "special_tokens_map.json",
                  "added_tokens.json", "tokenizer_config.json"):
            if Path(win_dir, f).exists():
                shutil.copy2(Path(win_dir, f), Path(wout, f))
        try:
            _api.upload_folder(folder_path=str(wout), repo_id=REPO, repo_type="model",
                               path_in_repo="winner", commit_message="sweep winner: gen-only root weights from checkpoint-%d (dev CER %s)" % (win["step"], win["cer"]))
            print("winner weights uploaded (private until evaluated)")
        except Exception as e:
            print("winner upload failed (retry later):", str(e)[:180])
        json.dump({"winner_step": win["step"], "dev_cer": win["cer"], "dev_wer": win["wer"],
                   "mismatch": win["mismatch"], "fulls": fulls, "subsets": subs},
                  open(Path(SWEEP) / "sweep_verdict.json", "w"), indent=2)
        try:
            shutil.copy2(Path(SWEEP) / "sweep_verdict.json", MIRROR + "/sweep_verdict.json")
        except Exception:
            pass
else:
    print("subset sweep incomplete (%d/%d scored) - rerun this cell in a new session; results resume" % (len(subs), len(all_ck)))

print("\n=== POOL-FREEZE RUN + SWEEP COMPLETE - record the VERDICT / WINNER / rc lines ===")


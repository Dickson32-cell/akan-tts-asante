# POOL HEAL CELL - attach to an orphaned trainer after the monitor cell died at step ~11000.
# Self-branching: if the trainer process is alive, re-attach (fixed mirror that CANNOT die:
# temp-file skip + copy-first-swap-later), wait for exit, then run convergence + Drive zip + push.
# If the trainer is gone, print exactly what to run (resume via the loader cell).
import glob, json, os, re, subprocess, time, shutil
from pathlib import Path

OUT = "/content/akan-vits-finetuned-pool"
LOG = "/content/train_akan_pool.log"   # REAL pool log name (train_akan.log was the 9h/freeze name)
MIRROR = "/content/drive/MyDrive/UG_TTS/pool_ckpts"
REPO = "Dickson32-cell/akan-twi-mms-pool"

pidtxt = Path("/content/train_pool_pid.txt")
pid = int(pidtxt.read_text()) if pidtxt.exists() else None

def _alive(p):
    try:
        os.kill(p, 0)
        # os.kill(pid,0) passes for zombies too; double-check via /proc state
        st = open(f"/proc/{p}/stat").read().split()[2]
        return st != "Z"
    except Exception:
        return False

def _ck_num(name):
    m = re.fullmatch(r"checkpoint-(\d+)", name)
    return int(m.group(1)) if m else None

def _ck_list(base):
    """checkpoint dirs only, sorted numerically; .partial/junk names are ignored."""
    out = []
    for p in glob.glob(base + "/checkpoint-*"):
        n = _ck_num(Path(p).name)
        if n is not None:
            out.append((n, p))
    return sorted(out)

def _loglines():
    try:
        return Path(LOG).read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return []

if not pid or not _alive(pid):
    print("=== TRAINER NOT RUNNING ===")
    loc = _ck_list(OUT); drv = _ck_list(MIRROR)
    print("local checkpoints:", [Path(p).name for _, p in loc[-3:]])
    print("drive mirrors  :", [Path(p).name for _, p in drv[-3:]])
    print("YOUR ACTION: re-run the LOADER cell (same one from Telegram) - it resumes from the newest checkpoint (local wins, else Drive).")
    raise SystemExit(0)

print("=== TRAINER ALIVE (pid", pid, ") - re-attaching with the FIXED mirror ===")
sz0 = Path(LOG).stat().st_size if Path(LOG).exists() else 0
time.sleep(20)
sz1 = Path(LOG).stat().st_size if Path(LOG).exists() else 0
print("log growth check:", sz0, "->", sz1, "| growing:", sz1 > sz0)

def mirror_newest():
    ck = _ck_list(OUT)
    if not ck:
        return None
    n, newest_path = ck[-1]
    newest = Path(newest_path).name
    tgt = MIRROR + "/" + newest
    if Path(tgt, ".mirror_done").exists():
        return newest
    # never publish a checkpoint the trainer is still writing or an incomplete one:
    _req = {"trainer_state.json", "pytorch_model.bin", "pytorch_model_1.bin"}
    _have = set(os.path.basename(f) for f in glob.glob(newest_path + "/*"))
    if any(x.startswith(".tmp") for x in _have) or not _req.issubset(_have):
        return newest  # mid-save or incomplete - retry on a later poll
    if not Path(MIRROR).exists():
        raise RuntimeError("Drive not mounted - run the loader cell which mounts Drive first")
    print("[MIRROR] saving", newest, "to Drive ...", flush=True)
    partial = tgt + ".partial"
    shutil.rmtree(partial, ignore_errors=True)
    shutil.copytree(newest_path, partial, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".tmp*"))
    # replace all old mirror dirs EXCEPT our in-flight partial (named newest+'.')
    for old in glob.glob(MIRROR + "/checkpoint-*"):
        if Path(old).name.startswith(newest + "."):
            continue
        shutil.rmtree(old, ignore_errors=True)
    os.rename(partial, tgt)
    try:
        shutil.copy2(LOG, MIRROR + "/train_akan_pool.log")
    except Exception:
        pass
    Path(tgt, ".mirror_done").write_text("mirrored")
    print("[MIRROR] done:", tgt, flush=True)
    return newest

# complete the killed checkpoint-11000 mirror first
try:
    print("catch-up mirror:", mirror_newest())
except Exception as e:
    print("[MIRROR] catch-up failed once (will retry in loop):", str(e)[:140])

print("attached - watching the trainer to completion (leave this tab open)")
lines = _loglines()
last = len(lines)
t0 = time.time()
while _alive(pid):
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
        mirror_newest()
    except Exception as e:
        print("[MIRROR] retry next poll:", str(e)[:140])
raw = _loglines()
done_ok = any("26265/26265" in l for l in raw) or any("train_runtime" in l for l in raw)
mins = round((time.time() - t0) / 60, 1)
print("\n=== TRAINER EXITED (watched %.1f min) ===" % mins)
if not done_ok:
    print("trainer ended WITHOUT completion markers - last lines:")
    for l in raw[-8:]:
        print("   ", l[:180])
    print("YOUR ACTION: re-run the LOADER cell - it resumes from the newest checkpoint (local wins, else Drive).")
    raise SystemExit("See the lines above.")
print("completion markers found - running convergence + push")

# ---- stage 8: convergence from tfevents ----
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
    json.dump({"verdict": locals().get("verdict", "CHECK MANUALLY"), "tags": tags},
              open(OUT + "/convergence_summary.json", "w"), indent=2)

# ---- stage 9: Drive zip + push to the POOL repo ----
os.makedirs("/content/drive/MyDrive/UG_TTS", exist_ok=True)
subprocess.run("zip -qr '/content/drive/MyDrive/UG_TTS/akan_tts_ckpt_pool.zip' " + OUT + " || true", shell=True)
from huggingface_hub import HfApi, upload_folder
api = HfApi()
try:
    api.create_repo(repo_id=REPO, repo_type="model", exist_ok=True)
    print("repo ready:", REPO)
except Exception as e:
    print("create_repo:", str(e)[:120])
try:
    api.upload_folder(folder_path=OUT, repo_id=REPO,
                      repo_type="model", commit_message="P1 pool run: 39.98h single-narrator, LR 1e-4, 15 epochs, seed 456 (all trainable)")
    print("PUSHED to", REPO, "(weights + artifacts)")
except Exception as e:
    print("push failed:", str(e)[:200])
    print("the checkpoint zip is safe in Drive; push again later from a new session")
print("\n=== POOL RUN COMPLETE - record the VERDICT lines for the final eval ===")

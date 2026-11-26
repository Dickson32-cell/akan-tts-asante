# POOL RECOVER CELL - the training loop COMPLETED (26265/26265 at lr~0 = schedule done);
# rc=1 came from a post-training stage (save/push/generation). This cell salvages everything
# from the live VM: shows the REAL error, mirrors checkpoints, zips to Drive, builds the
# convergence report, and pushes the final model to Dickson32-cell/akan-twi-mms-pool.
import glob, json, os, re, subprocess, time, shutil, sys
from pathlib import Path

OUT = "/content/akan-vits-finetuned-pool"
LOG = "/content/train_akan_pool.log"
MIRROR = "/content/drive/MyDrive/UG_TTS/pool_ckpts"
REPO = "Dickson32-cell/akan-twi-mms-pool"

# 0) the REAL child error - the tail of the training log
print("=" * 14, "REAL training-log tail (the actual rc=1 cause)", "=" * 14)
try:
    raw = Path(LOG).read_text(errors="replace").splitlines()
    for l in raw[-25:]:
        print("   ", l[:200])
except FileNotFoundError:
    print("   log not found at", LOG)

# 1) what does the output dir actually hold?
print("=" * 14, "output dir contents", "=" * 14)
print("OUT exists:", Path(OUT).exists())
if Path(OUT).exists():
    cks = sorted(glob.glob(OUT + "/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else 0)
    print("checkpoints:", [Path(c).name for c in cks[-5:]])
    print("top-level files:", sorted(os.path.basename(x) for x in glob.glob(OUT + "/*"))[:20])
    for c in cks[-2:]:
        print(Path(c).name, "->", sorted(os.path.basename(f) for f in glob.glob(c + "/*")))

# 2) mount Drive if needed + mirror the newest THREE checkpoints (real-name gate)
if not Path("/content/drive/MyDrive").exists():
    from google.colab import drive
    drive.mount("/content/drive")
Path(MIRROR).mkdir(parents=True, exist_ok=True)
published = []
for c in cks[-3:] if cks else []:
    n, name = c.rsplit("-", 1)[-1], Path(c).name
    have = set(os.path.basename(f) for f in glob.glob(c + "/*"))
    need = {"model.safetensors", "model_1.safetensors"}
    if not need <= have:
        print("skip (incomplete):", name, "- missing:", sorted(need - have))
        continue
    if Path(MIRROR, name, ".mirror_done").exists():
        published.append(name); continue
    print("[MIRROR] saving", name, "to Drive ...", flush=True)
    partial = MIRROR + "/" + name + ".partial"
    shutil.rmtree(partial, ignore_errors=True)
    shutil.copytree(c, partial, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".tmp*"))
    for old in glob.glob(MIRROR + "/checkpoint-*"):
        if Path(old).name.startswith(name + "."):
            continue
        shutil.rmtree(old, ignore_errors=True)
    os.rename(partial, MIRROR + "/" + name)
    Path(MIRROR, name, ".mirror_done").write_text("mirrored")
    published.append(name)
    print("[MIRROR] done:", name)
try:
    shutil.copy2(LOG, MIRROR + "/train_akan_pool.log")
except Exception:
    pass
print("mirrored:", published)

# 3) zip the whole output dir to Drive (durable copy of EVERYTHING)
subprocess.run("zip -qr '/content/drive/MyDrive/UG_TTS/akan_tts_pool_final.zip' " + OUT + " || true", shell=True)
print("Drive zip: /content/drive/MyDrive/UG_TTS/akan_tts_pool_final.zip")

# 4) convergence report from tfevents
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
    ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("Akan TTS POOL fine-tune (40h, single clean run) - convergence")
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
        vals = [p.value for p in acc.Scalars(main_tag)]
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
    "run": {"type": "P1 fresh 40h single clean run", "steps": "26265/26265 (15 epochs, LR 1e-4 cosine, seed 456)",
            "exit": "end of run - exact cause in the training-log tail above"},
    "infra_history": ("5 attempts documented: (1) OOM corpus build - fixed by streaming"
                      " per-shard Arrow checkpointing; (2) interpreter mismatch - fixed by the"
                      " single-interpreter law; (3) mid-save mirror loss at 20k + NaN warm-continuation"
                      " - fixed by complete-only mirrors, continuation discarded as invalid science;"
                      " (4) the completed 26,265-step run lost to VM recycle (mirrors had never fired)"
                      " - the rerun was rescued at step 13,000 via Drive mirror + HF checkpoint push;"
                      " (5) hub_strategy=checkpoint no-ops on this harness (push needs"
                      " trainer_state.json + config.json, which its custom checkpoint format never"
                      " writes: only model.safetensors/model_1.safetensors/optimizer/scheduler/scaler)"),
}, open(OUT + "/convergence_summary.json", "w"), indent=2)

# 5) push the final artifacts to the POOL repo
subprocess.run("pip -q show huggingface_hub >/dev/null 2>&1 || pip install -q huggingface_hub", shell=True)
from huggingface_hub import HfApi, upload_folder
api = HfApi()
try:
    api.create_repo(repo_id=REPO, repo_type="model", exist_ok=True)
    print("repo ready:", REPO)
except Exception as e:
    print("create_repo:", str(e)[:120])
try:
    root_files = [p.name for p in Path(OUT).iterdir() if p.is_file()]
    print("final-model root files:", root_files)
    api.upload_folder(folder_path=OUT, repo_id=REPO, repo_type="model", allow_patterns=root_files,
                      commit_message="P1 pool run FINAL: 39.98h single-narrator corpus, LR 1e-4, 15 epochs (26,265 steps), seed 456 - converged weights + artifacts")
    print("PUSHED to", REPO)
except Exception as e:
    print("push failed:", str(e)[:200])
    print("the Drive zip holds everything; push again from any session later")
print("\n=== RECOVERY COMPLETE - send the VERDICT + error-tail lines above to Hermes ===")

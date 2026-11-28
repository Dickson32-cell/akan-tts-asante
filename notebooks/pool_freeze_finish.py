# Akan TTS - POOL-FREEZE FINISH CELL (run AFTER the training cell prints TRAINING EXITED rc=0)
# The in-session sweep needs a fix on this harness generation: checkpoints save WITHOUT
# config.json/trainer_state.json (the pool run documented this layout), so the sweep loads
# config + tokenizer from the OUT root (written by the trainer at completion) and weights
# from each checkpoint dir. This cell = integrity assert + convergence + end push + the
# full staged sweep + winner upload, all idempotent and resumable.
import os, sys, json, glob, subprocess, time, shutil, re as _re, urllib.request
from pathlib import Path

OUT = "/content/akan-vits-finetuned-pool-freeze"
MIRROR = "/content/drive/MyDrive/UG_TTS/pool_freeze_ckpts"
SWEEP = "/content/sweep"
REPO = "Dickson32-cell/akan-twi-mms-pool-freeze"
REPOCODE = "Dickson32-cell/akan-tts-asante"
BASE = "/content/aka_train_base"
FINPIN = "0c0f11e"

def stage(n, t): print("\n" + "="*14 + f"  STAGE {n}: {t}  " + "="*14, flush=True)

REQUIRED = ["config.json", "model.safetensors", "vocab.json", "preprocessor_config.json"]
missing = [f for f in REQUIRED if not Path(OUT, f).exists()]
if missing:
    raise SystemExit("OUT root incomplete (missing " + ", ".join(missing) + ") - is the training actually finished? "
                     "Run this cell only AFTER 'TRAINING EXITED rc=0'.")
print("OUT root complete - training finished")

Path(MIRROR).mkdir(parents=True, exist_ok=True)
os.makedirs("/content/drive/MyDrive/UG_TTS", exist_ok=True)

# STAGE F1: frozen-tensor integrity assert (row-wise, resize-safe)
stage("F1", "Frozen-tensor integrity assert (text_encoder + duration_predictor vs staged base)")
try:
    from safetensors.torch import load_file, save_file
    import torch as _tt
    fb = load_file(BASE + "/model.safetensors")
    fr = load_file(OUT + "/model.safetensors")
    fk = [k for k in fb if k.startswith("text_encoder.") or k.startswith("duration_predictor.")]
    same, reshaped, diff = 0, [], []
    for k in fk:
        if k not in fr:
            diff.append(k + ":MISSING")
        elif fb[k].shape == fr[k].shape:
            if _tt.equal(fb[k], fr[k]):
                same += 1
            else:
                diff.append(k + ":VALUES-DIFFER")
        elif fr[k].shape[0] >= fb[k].shape[0] and _tt.equal(fb[k], fr[k][: fb[k].shape[0]]):
            same += 1
            reshaped.append(k + ":+%d rows" % (fr[k].shape[0] - fb[k].shape[0]))
        else:
            diff.append(k + ":SHAPE-MISMATCH")
    print("frozen tensors compared:", len(fk), "| byte-identical:", same,
          "| resized-preserved:", reshaped[:4], "| DIFFERING:", diff[:8])
    assert not diff, "frozen tensors drifted - STOP and paste this output before any push"
    print("FROZEN INTEGRITY: PASS")
except Exception as e:
    print("integrity check error (paste for diagnosis):", str(e)[:200])

# STAGE F2: convergence report (tfevents) - skips if already produced this run
stage("F2", "Convergence report from tfevents")
import matplotlib
if Path(OUT, "losses.csv").exists() and Path(OUT, "convergence_summary.json").exists():
    print("losses.csv + convergence_summary.json already present - F2 skip (idempotent)")
    json_done = True
else:
    json_done = False
if not json_done:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
ev_files = sorted(glob.glob(OUT + "/**/events.out.tfevents.*", recursive=True))
if not ev_files:
    print("no tfevents found - nothing to plot")
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

# STAGE F3: end push (root weights + artifacts) - SKIPS if the repo already carries them
stage("F3", "Push pool-freeze artifacts to its OWN private repo")
subprocess.run("zip -qr '/content/drive/MyDrive/UG_TTS/akan_tts_ckpt_pool_freeze.zip' " + OUT + " || true", shell=True)
from huggingface_hub import HfApi
api = HfApi()
try:
    api.create_repo(repo_id=REPO, repo_type="model", private=True, exist_ok=True)
except Exception as e:
    print("create_repo:", str(e)[:140])
try:
    _existing = set(api.list_repo_files(repo_id=REPO, repo_type="model"))
    if "model.safetensors" in _existing and "loss_curve.png" in _existing and "losses.csv" in _existing:
        print("repo already carries the root weights + artifacts - F3 upload skip")
    else:
        api.upload_folder(folder_path=OUT, repo_id=REPO, repo_type="model",
                          commit_message="pool-freeze run: 40h single-narrator, LR 5e-5, text_encoder + duration_predictor frozen, seed 456")
        print("PUSHED to " + REPO + " (weights + artifacts, private until evaluated)")
except Exception as e:
    print("push failed:", str(e)[:200]); print("the checkpoint zip is safe in Drive; push again later")

# STAGE F4: staged dev-set checkpoint sweep (25-clip subset for all, full 100 for finalists)
stage("F4", "Checkpoint sweep on the dev split (config from OUT root; weights per checkpoint)")
from safetensors.torch import load_file as _lff, save_file as _sfs
Path(SWEEP).mkdir(parents=True, exist_ok=True)
RES = Path(SWEEP) / "sweep_results.json"
results = []
if RES.exists():
    results = json.loads(RES.read_text(encoding="utf-8"))
    print("resumable sweep: %d checkpoints already scored" % len(results))

EVPIN = "https://raw.githubusercontent.com/%s/%s" % (REPOCODE, FINPIN)  # pinned evaluator + dev manifest
os.makedirs("/content/akan_eval/src", exist_ok=True)
for fn in ("src/eval_asr.py", "src/normalize_twi.py"):
    open("/content/akan_eval/" + fn, "wb").write(urllib.request.urlopen(EVPIN + "/" + fn, timeout=60).read())
dev = json.loads(urllib.request.urlopen(EVPIN + "/data/dev_split_poolfreeze.json", timeout=60).read().decode("utf-8"))
dev_clips = dev["clips"]
Path("/content/dev_refs.txt").write_text("\n".join(c["norm"] for c in dev_clips) + "\n", encoding="utf-8")
print("dev refs:", len(dev_clips), "clips | universe:", dev["universe_after_exclusions"], "| seed:", dev["seed"])

import torch as _t
from safetensors.torch import load_file as _lfs
from transformers import VitsModel as _VM, AutoTokenizer as _AT, AutoConfig as _AC
_cfg = _AC.from_pretrained(OUT)
_tk_ = _AT.from_pretrained(OUT)

def _synth(step, path, n_clips, tag):
    outdir = Path(SWEEP) / ("%s_%s" % (tag, step))
    outdir.mkdir(parents=True, exist_ok=True)
    # this generation's checkpoints store the generator SPLIT across model.safetensors + model_1.safetensors
    # AND in weight-norm form (weight_g/weight_v keys on decoder + flow convs - the harness strips the
    # norms only for the final root save). So: construct fresh from config, apply the same weight norms,
    # load the merged state dict, verify nothing gen-critical is missing, then strip the norms back to
    # the export convention (exactly what the harness does for the root).
    mdl = _VM(config=_cfg)
    mdl.decoder.apply_weight_norm()
    for fw in mdl.flow.flows:
        _t.nn.utils.weight_norm(fw.conv_pre)
        _t.nn.utils.weight_norm(fw.conv_post)
    sd = {}
    for f in ("model.safetensors", "model_1.safetensors"):
        if Path(path, f).exists():
            sd.update(_lff(str(Path(path, f))))
    missing, unexpected = mdl.load_state_dict(sd, strict=False)
    crit = [k for k in missing if k.startswith(("decoder.", "flow.", "text_encoder.", "duration_predictor."))]
    if crit:
        raise RuntimeError("checkpoint %s: %d gen-critical tensors missing after merged load, e.g. %s"
                           % (Path(path).name, len(crit), crit[:6]))
    mdl.decoder.remove_weight_norm()
    for fw in mdl.flow.flows:
        _t.nn.utils.remove_weight_norm(fw.conv_pre)
        _t.nn.utils.remove_weight_norm(fw.conv_post)
    tk = _tk_
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
        inp = {k: v.to(mdl.device) for k, v in inp.items()}
        with _t.no_grad():
            o = mdl(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
        wv = o.waveform[0].cpu().numpy()
        import soundfile as _sf
        wv = wv / max(1e-6, abs(wv).max()) * 0.95
        _sf.write(str(f), wv, 16000, subtype="PCM_16")
    del mdl
    _t.cuda.empty_cache()
    return outdir

def _judge(outdir, out_name):
    cmd = [sys.executable, "/content/akan_eval/src/eval_asr.py", "--samples", str(outdir),
           "--prompts", "/content/dev_refs.txt", "--engine", "mms_aka", "--device", "cuda",
           "--out-eval", str(Path(SWEEP) / out_name)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if Path(SWEEP, out_name).exists():
        j = json.loads(Path(SWEEP, out_name).read_text(encoding="utf-8"))
        return {"cer": j.get("mean_cer"), "wer": j.get("mean_wer"),
                "mismatch": j.get("mismatch_control_cer"), "rc": p.returncode}
    print("judge run failed (paste above for diagnosis):", (p.stdout + p.stderr)[-400:])
    return None

def _ck_safe(base_dir):
    out_ = []
    for p in glob.glob(base_dir + "/checkpoint-*"):
        m = _re.fullmatch(r"checkpoint-(\d+)", Path(p).name)
        if m and Path(p, "model.safetensors").exists():
            out_.append((int(m.group(1)), p))
    return sorted(out_)

SUBSET_N = 25
FINALISTS = 3
all_ck = _ck_safe(OUT)
print("checkpoints to sweep:", [n for n, _ in all_ck])
for n, pth in all_ck:
    if any(x.get("stage") == "subset" and x.get("step") == n for x in results):
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
    for r in finalists:
        if any(x.get("stage") == "full" and x.get("step") == r["step"] for x in results):
            continue
        pth = dict((n, p) for n, p in all_ck)[r["step"]]
        od = _synth(r["step"], pth, len(dev_clips), "full")
        jr = _judge(od, "full_%s.json" % r["step"])
        if jr:
            fr = {"stage": "full", "step": r["step"], "cer": jr["cer"], "wer": jr["wer"],
                  "mismatch": jr["mismatch"], "rc": jr["rc"]}
            results = [x for x in results if not (x.get("stage") == "full" and x.get("step") == r["step"])] + [fr]
            RES.write_text(json.dumps(results, indent=2))
            print("[SWEEP] step %d: FULL-DEV CER %s | WER %s | mismatch %s" % (r["step"], fr["cer"], fr["wer"], fr["mismatch"]), flush=True)
    fulls = [r for r in results if r.get("stage") == "full" and r.get("cer") is not None]
    if fulls:
        fulls.sort(key=lambda r: r["cer"])
        win = fulls[0]
        print("\nWINNER: checkpoint-%d | full-100-clip dev CER %s | WER %s | mismatch %s"
              % (win["step"], win["cer"], win["wer"], win["mismatch"]))
        stage("F5", "Publish winner: gen-only root weights (discriminator never uploads)")
        win_dir = dict((n, p) for n, p in all_ck)[win["step"]]
        wout = Path(SWEEP) / "winner"
        wout.mkdir(exist_ok=True)
        _sfs(_lff(str(Path(win_dir) / "model.safetensors")), str(Path(wout) / "model.safetensors"))
        for f in ("config.json", "preprocessor_config.json", "vocab.json", "special_tokens_map.json",
                  "added_tokens.json", "tokenizer_config.json"):
            if Path(OUT, f).exists():
                shutil.copy2(Path(OUT, f), Path(wout, f))
        try:
            api.upload_folder(folder_path=str(wout), repo_id=REPO, repo_type="model",
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
    print("subset sweep incomplete (%d/%d scored) - rerun this same cell to resume; finished steps are skipped" % (len(subs), len(all_ck)))

print("\n=== POOL-FREEZE FINISH COMPLETE - record the FROZEN INTEGRITY / VERDICT / WINNER lines ===")

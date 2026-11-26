# POOL RESUME CELL v1 - the trainer is ALIVE but BLOCKED (its output buffer filled after the
# monitor died). This cell: kills the frozen trainer, injects the resume-state files the custom
# checkpoint format never writes (trainer_state.json + config.json + torch-format weight copies),
# enables resume in the config, relaunches training FROM checkpoint-13000 with output going
# straight to a NEW log file (no pipes, cannot re-freeze), and watches the first minutes.
import json, os, re, glob, shutil, subprocess, sys, time
from pathlib import Path

OUT = "/content/akan-vits-finetuned-pool"
CK = OUT + "/checkpoint-13000"
CFGJ = "/content/finetune_akan_pool.json"
HR = "/content/finetune-hf-vits"
NEWLOG = "/content/train_akan_pool_resume.log"
STEPS = "13000"
EPOCH = "7.368"   # 13000 of 1764.6 steps per epoch
MAXS = "26265"

print("=" * 12, "stage 0: environment", "=" * 12)
for f in (CK + "/model.safetensors", CK + "/model_1.safetensors", OUT + "/config.json",
          "/content/tts_data/dataset_dict.json", HR + "/run_vits_finetuning.py", CFGJ):
    p = Path(f)
    print(("OK  " if p.exists() else "MISSING ") + f)
assert Path(CK + "/model.safetensors").exists() and Path(CK + "/model_1.safetensors").exists(), "checkpoint-13000 incomplete - aborting"

print("=" * 12, "stage 1: kill the blocked trainer", "=" * 12)
pidtxt = Path("/content/train_pool_pid.txt")
blocked = pidtxt.read_text().strip() if pidtxt.exists() else ""
if blocked and Path("/proc/" + blocked).exists():
    subprocess.run(["kill", "-9", blocked])
    print("killed blocked trainer pid", blocked)
for pat in ("run_vits_finetuning", "accelerate.commands.launch"):
    subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
time.sleep(4)
print("cleared. remaining accelerates:", subprocess.run(["pgrep", "-f", "accelerate"], capture_output=True).stdout.decode().strip() or "none")

print("=" * 12, "stage 2: inject resume-state files", "=" * 12)
state = {
    "global_step": int(STEPS), "max_steps": int(MAXS), "num_train_epochs": 15,
    "epoch": float(EPOCH), "log_history": [], "best_metric": None, "best_model_checkpoint": None,
    "is_local_process_zero": True, "is_world_process_zero": True, "stateful_callbacks": [],
}
Path(CK, "trainer_state.json").write_text(json.dumps(state, indent=2))
print("trainer_state.json written: global_step", STEPS)
shutil.copy2(OUT + "/config.json", CK + "/config.json")
print("config.json copied from run root")
from safetensors.torch import load_file as st_load
import torch
for src, dst in ((CK + "/model.safetensors", CK + "/pytorch_model.bin"),
                 (CK + "/model_1.safetensors", CK + "/pytorch_model_1.bin")):
    if not Path(dst).exists():
        torch.save(st_load(src), dst)
        print("torch-format copy written:", os.path.basename(dst))
print("checkpoint-13000 now carries:", sorted(os.path.basename(x) for x in glob.glob(CK + "/*")))

print("=" * 12, "stage 3: enable resume in the config", "=" * 12)
cfg = json.loads(Path(CFGJ).read_text())
cfg["resume_from_checkpoint"] = "latest"   # STRING, not bool - harness law
Path(CFGJ + ".bak_fresh").write_text(json.dumps(cfg, indent=2))  # keep a fresh-run copy
Path(CFGJ).write_text(json.dumps(cfg, indent=2))
print("resume_from_checkpoint enabled (backup at finetune_akan_pool.json.bak_fresh)")

print("=" * 12, "stage 4: relaunch (direct file redirect - cannot re-block)", "=" * 12)
subprocess.run(f"rm -f {NEWLOG}", shell=True)
launch = (f"cd {HR} && nohup {sys.executable} -m accelerate.commands.launch --num_processes 1 "
          f"--mixed_precision fp16 run_vits_finetuning.py {CFGJ} > {NEWLOG} 2>&1 & echo $! > /content/train_pool_pid.txt")
subprocess.run(launch, shell=True)
time.sleep(8)
newpid = Path("/content/train_pool_pid.txt").read_text().strip()
print("relaunched, pid:", newpid)

print("=" * 12, "stage 5: watch the first minutes", "=" * 12)
ok, nan = False, False
for i in range(24):
    time.sleep(15)
    txt = Path(NEWLOG).read_text(errors="replace")
    lines = txt.splitlines()
    steps = [l for l in lines if "/26265 [" in l]
    if "nan" in txt.lower():
        nan = True
    if "resuming from checkpoint" in txt or "Loaded model from" in txt or any("checkpoint-13000" in l for l in lines):
        ok = True
    last = steps[-1][:120] if steps else (lines[-1][:120] if lines else "(no output yet)")
    print(f"poll {i+1}: " + last)
    if nan:
        break
    if i >= 5 and steps:
        m = re.search(r"(\d+)/26265", steps[-1])
        if m:
            print("FIRST STEP MARKER:", m.group(1))
            ok = True
            break
print("=" * 8)
if nan:
    print("WARNING: 'nan' appeared in the output - if the newest step losses read nan, we kill the run and fall back to a fresh one")
elif ok:
    print("RESUME RUNNING - training continues from step 13000; leave the tab open; do NOT press stop on this cell")
else:
    print("NO STEP LINES YET - keep this output and send it for the next adjustment")
print("=== RESUME CELL DONE ===")

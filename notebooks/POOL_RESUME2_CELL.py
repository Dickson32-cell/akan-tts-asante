# POOL RESUME2 CELL - fresh-VM recovery: re-stage (corpus + base) via the proven stage-1 file,
# pull checkpoint-13000 back from Drive, re-inject the resume-state files, enable resume in the
# config, relaunch training FROM step 13000 into a NEW direct-file log, and watch the first minutes.
import json, os, glob, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

OUT = "/content/akan-vits-finetuned-pool"
MIRROR = "/content/drive/MyDrive/UG_TTS/pool_ckpts"
CFGJ = "/content/finetune_akan_pool.json"
HR = "/content/finetune-hf-vits"
LOG = "/content/train_akan_pool_resume2.log"
STEPS, EPOCH, MAXS = 13000, "7.368", 26265

print("=" * 12, "stage 0: VM diagnosis", "=" * 12)
print("VM uptime (hours):", round(float(open("/proc/uptime").read().split()[0]) / 3600, 1))
print("local checkpoints:", [os.path.basename(c) for c in glob.glob(OUT + "/checkpoint-*")])
print("tts_data:", os.path.exists("/content/tts_data/dataset_dict.json"), "| base:", os.path.exists("/content/aka_train_base/config.json"))

print("=" * 12, "stage 1..6: re-staging via the proven stage-1 file (skips what already exists)", "=" * 12)
P1 = "https://raw.githubusercontent.com/Dickson32-cell/akan-tts-asante/2b25bf7/notebooks/pool_stage1.py"
exec(compile(urllib.request.urlopen(P1, timeout=60).read().decode("utf-8"), "pool_stage1", "exec"), globals())
# stage1 exits with SystemExit: NO GPU if quota is dry - that print IS the diagnosis

print("=" * 12, "stage 7a: secure checkpoint-13000 locally", "=" * 12)
ck_local = OUT + "/checkpoint-" + str(STEPS)
state_injected = os.path.exists(ck_local + "/trainer_state.json")
if not state_injected:
    src = MIRROR + "/checkpoint-" + str(STEPS)
    assert os.path.exists(src + "/model.safetensors"), "Drive checkpoint-13000 missing - send this output"
    shutil.rmtree(ck_local, ignore_errors=True)
    shutil.copytree(src, ck_local)
    print("pulled checkpoint-13000 from Drive")
from safetensors.torch import load_file as st_load
import torch
for src2, dst in ((ck_local + "/model.safetensors", ck_local + "/pytorch_model.bin"),
                  (ck_local + "/model_1.safetensors", ck_local + "/pytorch_model_1.bin")):
    if not os.path.exists(dst):
        torch.save(st_load(src2), dst)
        print("torch-format copy:", os.path.basename(dst))
if not os.path.exists(ck_local + "/trainer_state.json"):
    Path(ck_local, "trainer_state.json").write_text(json.dumps({
        "global_step": STEPS, "max_steps": MAXS, "num_train_epochs": 15,
        "epoch": float(EPOCH), "log_history": [], "best_metric": None,
        "best_model_checkpoint": None, "is_local_process_zero": True,
        "is_world_process_zero": True, "stateful_callbacks": []}, indent=2))
    print("trainer_state.json injected: global_step", STEPS)
if not os.path.exists(ck_local + "/config.json"):
    shutil.copy2(OUT + "/config.json", ck_local + "/config.json")
    print("config.json copied from run root")

print("=" * 12, "stage 7b: enable resume in the config", "=" * 12)
cfg = json.loads(Path(CFGJ).read_text())
cfg["resume_from_checkpoint"] = "latest"   # STRING law
Path(CFGJ).write_text(json.dumps(cfg, indent=2))
print("resume_from_checkpoint: latest")

print("=" * 12, "stage 7c: kill strays + relaunch (direct-file log, cannot re-block)", "=" * 12)
for pat in ("run_vits_finetuning", "accelerate.commands.launch"):
    subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
time.sleep(3)
subprocess.run(f"rm -f {LOG}", shell=True)
launch = (f"cd {HR} && nohup {sys.executable} -m accelerate.commands.launch --num_processes 1 "
          f"--mixed_precision fp16 run_vits_finetuning.py {CFGJ} > {LOG} 2>&1 & echo $! > /content/train_pool_pid.txt")
subprocess.run(launch, shell=True)
time.sleep(8)
print("relaunched, pid:", Path("/content/train_pool_pid.txt").read_text().strip())

print("=" * 12, "stage 8: watch the first minutes (true-NaN pattern only)", "=" * 12)
import re
adv = False
for i in range(24):
    time.sleep(15)
    txt = Path(LOG).read_text(errors="replace")
    lines = txt.splitlines()
    steps = [l for l in lines if f"/{MAXS} [" in l]
    nums = [int(m) for l in steps for m in re.findall(rf"(\d+)/{MAXS}", l)]
    nan = len(re.findall(r"step_loss=nan|step_loss_disc=nan", txt))
    last = steps[-1][:120] if steps else (lines[-1][:120] if lines else "(no output yet)")
    print(f"poll {i+1}: " + last)
    if nums and max(nums) > STEPS:
        adv = True
        break
    if nan:
        print("WARNING: true NaN losses appeared - kill this run and fall back to fresh")
        break
print("=" * 8)
print("RESUME2 OK - training advanced past step", STEPS if adv else "?",
      "- leave the tab open; recovery cell (0dc93e0) runs when the counter reaches", MAXS)

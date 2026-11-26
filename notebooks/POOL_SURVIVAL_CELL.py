# POOL SURVIVAL CHECK - defensive (never crashes); tells us which recovery branch to take
import os, glob, time
def chk(p):
    return os.path.exists(p)
print("VM uptime (hours):", round(float(open("/proc/uptime").read().split()[0]) / 3600, 1))
for name, p in [("OUT dir", "/content/akan-vits-finetuned-pool"),
                ("tts_data corpus", "/content/tts_data/dataset_dict.json"),
                ("staged base", "/content/aka_train_base/config.json"),
                ("old training log", "/content/train_akan_pool.log"),
                ("resume log", "/content/train_akan_pool_resume.log"),
                ("pid file", "/content/train_pool_pid.txt")]:
    print(("OK      " if chk(p) else "MISSING ") + name)
cks = sorted(glob.glob("/content/akan-vits-finetuned-pool/checkpoint-*"), key=lambda x: int(x.rsplit("-", 1)[-1]) if x.rsplit("-", 1)[-1].isdigit() else 0)
print("local checkpoints:", [os.path.basename(c) for c in cks])
if cks:
    newest = cks[-1]
    have = set(os.path.basename(f) for f in glob.glob(newest + "/*"))
    print("newest has state files:", "trainer_state.json" in have, "| config:", "config.json" in have)
    root = "/content/akan-vits-finetuned-pool"
    print("final model at root:", chk(root + "/model.safetensors"), chk(root + "/model_1.safetensors"))

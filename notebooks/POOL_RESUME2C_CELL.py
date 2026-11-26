# RESUME2C FIX-UP - completes checkpoint-13000 (config.json from the staged base) and mirrors
# the injected state files into the Drive copy, so whichever copy the loader's resume ladder
# picks (local or Drive) is complete. No launch here - the standard loader runs after this.
import json, os, glob, shutil
from pathlib import Path

OUT = "/content/akan-vits-finetuned-pool"
MIRROR = "/content/drive/MyDrive/UG_TTS/pool_ckpts"
BASE_CFG = "/content/aka_train_base/config.json"
CK = OUT + "/checkpoint-13000"

# 1) local checkpoint: verify it survived the crashed cell
have = set(os.path.basename(f) for f in glob.glob(CK + "/*")) if os.path.exists(CK) else set()
print("local checkpoint-13000 files:", sorted(have))
assert {"model.safetensors", "model_1.safetensors", "optimizer.bin", "optimizer_1.bin",
        "trainer_state.json", "pytorch_model.bin", "pytorch_model_1.bin"} <= have, \
    "checkpoint incomplete - run RESUME2 again to rebuild, then re-run this cell"

# 2) add config.json from the staged base (the fresh VM has no OUT-root config yet)
if "config.json" not in have:
    shutil.copy2(BASE_CFG, CK + "/config.json")
    print("config.json copied from aka_train_base")
else:
    print("config.json already present")

# 3) sync the injected state into the DRIVE copy (in case the ladder prefers Drive)
drv = MIRROR + "/checkpoint-13000"
if os.path.exists(drv):
    for f in ("trainer_state.json", "config.json", "pytorch_model.bin", "pytorch_model_1.bin"):
        src, dst = CK + "/" + f, drv + "/" + f
        if os.path.exists(dst):
            os.remove(dst)
        shutil.copy2(src, dst)
    print("Drive copy of checkpoint-13000 now carries the same state files")
else:
    shutil.rmtree(MIRROR + "/checkpoint-13000.partial", ignore_errors=True)
    shutil.copytree(CK, MIRROR + "/checkpoint-13000.partial")
    os.rename(MIRROR + "/checkpoint-13000.partial", drv)
    print("Drive had no checkpoint-13000 - full copy written")
print("drive copy files:", sorted(os.path.basename(x) for x in glob.glob(drv + "/*")))
print("=== FIX-UP DONE - now run the standard loader cell ===")

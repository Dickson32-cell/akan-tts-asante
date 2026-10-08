# Akan TTS - STAGES 1-5 (environment: GPU, harness, login, corpus, base checkpoint)
# Fetched and exec'ed by the loader. Idempotent: every stage skips if already done.
# NOTE: version checks NEVER import datasets in-kernel (importlib.metadata only) -
# in-kernel re-import poisons pyarrow's extension registry (ArrowKeyError, Oct 2026).
import os, sys, json, glob, subprocess, time, urllib.request
from pathlib import Path

def stage(n, t): print("\n" + "="*14 + f"  STAGE {n}: {t}  " + "="*14, flush=True)

def shell(cmd, must=True, cwd=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0 and must:
        print("COMMAND FAILED:", cmd)
        print((r.stdout or "")[-1200:]); print((r.stderr or "")[-1200:])
        raise SystemExit("Send the lines above to Hermes in Telegram.")
    return r

# stage 1: GPU
stage(1, "GPU check")
import torch
if not torch.cuda.is_available():
    raise SystemExit("NO GPU - Runtime > Change runtime type > T4 GPU > save > run this cell again.")
print("GPU OK:", torch.cuda.get_device_name(0))

# stage 2: harness + pins (version check via METADATA - no in-kernel datasets import)
stage(2, "Harness + versions")
import importlib.metadata as md
_v_t, _v_d = md.version("transformers"), md.version("datasets")
print("versions (metadata): transformers", _v_t, "| datasets", _v_d)
need_install = not (_v_t == "4.49.0" and _v_d.startswith("3.6"))
harness_ok = Path("/content/finetune-hf-vits/run_vits_finetuning.py").exists()
mono_ok = bool(glob.glob("/content/finetune-hf-vits/monotonic_align/monotonic_align/*.so")) or \
          Path("/content/finetune-hf-vits/monotonic_align/monotonic_align/core.py").exists()
if need_install or not harness_ok or not mono_ok:
    shell("pip install -q 'transformers==4.49.0'")
    shell("pip install -q 'datasets[audio]>=3.2,<4.0' librosa soundfile accelerate", must=False)
    if not harness_ok:
        shell("git clone https://github.com/ylacombe/finetune-hf-vits /content/finetune-hf-vits")
    shell("pip install -q -r /content/finetune-hf-vits/requirements.txt", must=False)
    shell("mkdir -p /content/finetune-hf-vits/monotonic_align/monotonic_align")
    shell("cp -n /content/finetune-hf-vits/monotonic_align/core.py /content/finetune-hf-vits/monotonic_align/monotonic_align/core.py || true")
    shell("cd /content/finetune-hf-vits/monotonic_align && python setup.py build_ext --inplace", must=False)
print("harness ready:", harness_ok, "| pins:", _v_t, _v_d)

# stage 3: HF login (token probe via whoami; login only if needed)
stage(3, "Hugging Face login")
from huggingface_hub import HfApi
api = HfApi()
try:
    me = api.whoami()
    print("already logged in as:", me["name"])
except Exception:
    from huggingface_hub import notebook_login
    notebook_login()
    api = HfApi()
    try:
        me = api.whoami(); print("logged in as:", me["name"])
    except Exception:
        raise SystemExit("Login incomplete - run this cell again and finish the sign-in.")

# stage 4: corpus (skip if /content/tts_data exists)
stage(4, "Corpus (Drive > unzip > stream selected audio)")
if Path("/content/tts_data").exists():
    print("tts_data present > skip")
else:
    from google.colab import drive
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")
    shell("mkdir -p /content/work && unzip -o -q '/content/drive/MyDrive/UG_TTS/tts_data.zip' -d /content/work")
    base = "https://raw.githubusercontent.com/Dickson32-cell/akan-tts-asante/master/src/"
    for f in ["normalize_twi.py", "extract_asante_colab.py"]:
        Path("/content/" + f).write_text(
            urllib.request.urlopen(base + f, timeout=60).read().decode("utf-8"), encoding="utf-8")
    r = shell("python /content/extract_asante_colab.py --manifest /content/work/selected.jsonl --out /content/tts_data", must=False)
    if r.returncode != 0:
        print((r.stdout or "")[-900:]); print((r.stderr or "")[-900:])
        raise SystemExit("corpus build failed - send lines above to Hermes")

# stage 5: base training checkpoint (skip if built)
stage(5, "Training base (generator 'aka' + Meta discriminator)")
if Path("/content/aka_train_base").exists() and glob.glob("/content/aka_train_base/*.safetensors"):
    print("base checkpoint present > skip")
else:
    r = shell("python /content/finetune-hf-vits/convert_original_discriminator_checkpoint.py --language_code aka --pytorch_dump_folder_path /content/aka_train_base",
              cwd="/content/finetune-hf-vits", must=False)
    if r.returncode != 0:
        print((r.stdout or "")[-900:]); print((r.stderr or "")[-900:])
        raise SystemExit("discriminator conversion failed - send lines above to Hermes")
    print("base ready:", sorted(os.path.basename(x) for x in glob.glob("/content/aka_train_base/*")))

print("\nSTAGES 1-5 COMPLETE - the loader now runs stages 6-9 (patches > training > convergence > push)")
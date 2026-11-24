# Akan TTS - STAGES 1-5 for the POOL RUN (40h full-pool training, P1)
# Differences vs freeze_stage1 (both verified patterns):
#   stage 4: corpus zip = tts_data_40h.zip (the 39.98h manifest) instead of tts_data.zip
#   stage 5: training base = the ORIGINAL Meta base checkpoint (facebook/mms-tts-aka),
#            because P1 is the clean data-scaling experiment run from the true base.
# Idempotent: each stage skips if already done in this session.
import os, sys, json, glob, subprocess, time, urllib.request
from pathlib import Path

def stage(n, t): print("\n" + "="*14 + f"  STAGE {n}: {t}  " + "="*14, flush=True)

def shell(cmd, must=True, cwd=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0 and must:
        print("COMMAND FAILED:", cmd)
        print((r.stdout or "")[-1200:]); print((r.stderr or "")[-1200:])
        raise SystemExit("The lines above state the failing stage and its fix.")
    return r

# stage 1: GPU
stage(1, "GPU check")
import torch
if not torch.cuda.is_available():
    raise SystemExit("NO GPU - Runtime > Change runtime type > T4 GPU > save > run this cell again.")
print("GPU OK:", torch.cuda.get_device_name(0))

# stage 2: harness + pins - ONE interpreter law: install AND verify with
# sys.executable -m pip (plain 'pip' can target a different python3.13 env than
# the launcher's interpreter; that dual-env mismatch broke the pool run once).
stage(2, "Harness + versions (single-interpreter pins)")
print("python interpreter:", sys.executable)
def _pins_of(exe):
    import subprocess as _sp
    r = _sp.run([exe, "-c",
                 "import importlib.metadata as m; print(m.version('transformers'), m.version('datasets'))"],
                capture_output=True, text=True, timeout=120)
    return r.stdout.split() if r.returncode == 0 else []
_v_t, _v_d = _pins_of(sys.executable) or ["?", "?"]
print("versions (the launcher interpreter): transformers", _v_t, "| datasets", _v_d)
need_install = not (_v_t == "4.49.0" and _v_d == "3.2.0")
harness_ok = Path("/content/finetune-hf-vits/run_vits_finetuning.py").exists()
mono_ok = bool(glob.glob("/content/finetune-hf-vits/monotonic_align/monotonic_align/*.so")) or \
          Path("/content/finetune-hf-vits/monotonic_align/monotonic_align/core.py").exists()
if need_install or not harness_ok or not mono_ok:
    shell(sys.executable + " -m pip install -q 'transformers==4.49.0'")
    shell(sys.executable + " -m pip install -q 'datasets==3.2.0' librosa soundfile accelerate", must=False)
    if not harness_ok:
        shell("git clone https://github.com/ylacombe/finetune-hf-vits /content/finetune-hf-vits")
    shell(sys.executable + " -m pip install -q -r /content/finetune-hf-vits/requirements.txt", must=False)
    shell("mkdir -p /content/finetune-hf-vits/monotonic_align/monotonic_align")
    shell("cp -n /content/finetune-hf-vits/monotonic_align/core.py /content/finetune-hf-vits/monotonic_align/monotonic_align/core.py || true")
    shell("cd /content/finetune-hf-vits/monotonic_align && " + sys.executable + " setup.py build_ext --inplace", must=False)
_v_t2, _v_d2 = _pins_of(sys.executable) or ["?", "?"]
print("harness ready:", harness_ok and mono_ok, "| pins verified after install:", _v_t2, _v_d2)
if not (_v_t2 == "4.49.0" and _v_d2 == "3.2.0"):
    raise SystemExit("PIN FAILED on the launcher interpreter (" + sys.executable +
                     "): transformers " + _v_t2 + " / datasets " + _v_d2 +
                     " - NOTHING IS WRONG with the project; the environment needs the pins. "
                     "Rerun this cell once more (it re-installs); if it still fails, "
                     "Runtime > Restart session, then rerun this cell.")
print("single-interpreter pin law: OK")

# stage 3: HF login - Colab secret HF_TOKEN first (enables background/Run-all),
# then stored credentials, then the interactive widget as last resort.
stage(3, "Hugging Face login")
from huggingface_hub import HfApi
api = HfApi()
def _who():
    try:
        me = api.whoami(); print("logged in as:", me["name"]); return True
    except Exception:
        return False
if _who():
    pass
else:
    _t = None
    try:
        from google.colab import userdata
        _t = userdata.get("HF_TOKEN")
        print("using Colab secret: HF_TOKEN")
    except Exception:
        try:
            from huggingface_hub import get_token
            _t = get_token()
        except Exception:
            _t = None
    if _t:
        from huggingface_hub import login as _login
        _login(token=_t)
        if not _who():
            raise SystemExit("Colab secret HF_TOKEN did not authenticate - check the token has WRITE access.")
    else:
        from huggingface_hub import notebook_login
        notebook_login()
        if not _who():
            raise SystemExit("Login incomplete - run this cell again and finish the sign-in.")

# stage 4: corpus (VALIDATE, not mere exists) - THE POOL MANIFEST (39.98h)
stage(4, "Corpus (Drive > unzip tts_data_40h.zip > stream selected audio)")
def _dataset_dir_valid(pth):
    base = Path(pth)
    return base.joinpath("dataset_dict.json").exists() and \
           ((base / "train" / "state.json").exists() or (base / "train" / "dataset_info.json").exists())
if _dataset_dir_valid("/content/tts_data"):
    print("tts_data present AND valid > skip")
else:
    if Path("/content/tts_data").exists():
        import shutil as _sh
        _sh.rmtree("/content/tts_data", ignore_errors=True)
        print("tts_data dir INVALID (partial build) > cleaned - rebuilding from shard checkpoints")

    from google.colab import drive
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")
    _z = "/content/drive/MyDrive/UG_TTS/tts_data_40h.zip"
    if not Path(_z).exists():
        raise SystemExit("Drive file missing: " + _z + " - upload E:/UG-TTS-Task/02_data/tts_data_40h.zip into the Drive folder UG_TTS, then run this cell again.")
    shell("mkdir -p /content/work && unzip -o -q '" + _z + "' -d /content/work")
    _n = sum(1 for _l in open("/content/work/selected.jsonl", encoding="utf-8") if _l.strip())
    print("pool manifest rows:", _n)
    assert _n == 28434, "POOL MANIFEST MISMATCH - expected 28434 rows (28234 train + 200 eval). STOPPING before any training."
    base = "https://raw.githubusercontent.com/Dickson32-cell/akan-tts-asante/6139b1a/src/"
    for f in ["normalize_twi.py", "extract_asante_colab.py"]:
        Path("/content/" + f).write_text(
            urllib.request.urlopen(base + f, timeout=60).read().decode("utf-8"), encoding="utf-8")
    r = shell("python /content/extract_asante_colab.py --manifest /content/work/selected.jsonl --out /content/tts_data", must=False)
    if r.returncode != 0:
        print((r.stdout or "")[-900:]); print((r.stderr or "")[-900:])
        raise SystemExit("corpus build failed - Review the output above - the stopped stage names its own fix")

# stage 5: training base = ORIGINAL Meta base checkpoint (the clean data-scaling anchor)
stage(5, "Training base = ORIGINAL Meta base (facebook/mms-tts-aka)")
def _base_ok():
    return Path("/content/aka_train_base").exists() and glob.glob("/content/aka_train_base/*.safetensors") \
           and Path("/content/aka_train_base/preprocessor_config.json").exists()
if _base_ok():
    print("pool base present > skip")
else:
    from huggingface_hub import snapshot_download
    if not (Path("/content/aka_train_base").exists() and glob.glob("/content/aka_train_base/*.safetensors")):
        snapshot_download(repo_id="facebook/mms-tts-aka",
                          local_dir="/content/aka_train_base",
                          allow_patterns=["*.json", "*.txt", "*.safetensors"])
    # the harness's vendored VitsFeatureExtractor.from_pretrained hard-errors when
    # preprocessor_config.json is absent (hub AND local, this transformers stack).
    # Content = the values the two COMPLETED runs themselves used (original repo +
    # freeze repo carry this same file): model-native 16000 Hz + class defaults.
    _pcfg = {
        "feature_extractor_type": "VitsFeatureExtractor",
        "feature_size": 80, "hop_length": 256, "max_wav_value": 32768.0,
        "n_fft": 1024, "padding_side": "right", "padding_value": 0.0,
        "return_attention_mask": False, "sampling_rate": 16000,
    }
    json.dump(_pcfg, open("/content/aka_train_base/preprocessor_config.json", "w"), indent=2)
    _ok = json.load(open("/content/aka_train_base/preprocessor_config.json"))["sampling_rate"] == 16000
    print("pool base ready:", sorted(os.path.basename(x) for x in glob.glob("/content/aka_train_base/*")),
          "| preprocessor_config written:", _ok)
    if not _ok:
        raise SystemExit("preprocessor config failed verification - Review the output above")

print("\nSTAGES 1-5 COMPLETE - loader continues with stages 6-9 (patches > pool training with Drive checkpoints > convergence > push)")

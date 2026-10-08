# ═══════════════ RUN-EVERYTHING — THE ONLY CELL YOU RUN ═══════════════
# Press play on THIS cell. Every time. Any number of times. It knows the order:
# fresh session → runs everything; interrupted → continues; mid-training → just monitors.
# (The cells below are kept as reference - do not run them.)
import os, sys, json, glob, subprocess, time, shutil, urllib.request
from pathlib import Path

def stage(n, t): print("\n" + "="*14 + f"  STAGE {n}: {t}  " + "="*14)

def shell(cmd, must=True, cwd=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0 and must:
        print("COMMAND FAILED:", cmd); print(r.stdout[-1500:] if r.stdout else ""); print(r.stderr[-1500:])
        raise SystemExit("Send the lines above to Hermes in Telegram.")
    return r

# ── S1: GPU ──────────────────────────────────────────────────────────
stage(1, "GPU check")
import torch
if not torch.cuda.is_available():
    raise SystemExit("NO GPU — Runtime menu → 'Change runtime type' → T4 GPU → save → run this cell again.")
print("GPU OK:", torch.cuda.get_device_name(0))

# ── S2: harness + pin (skips when already done) ──────────────────────
stage(2, "Harness + versions")
import importlib.metadata as md
need_install = True
try:
    if md.version("transformers").startswith("4.49"):
        need_install = Path("/content/finetune-hf-vits/run_vits_finetuning.py").exists()
        if not need_install: print("transformers 4.49 ok; harness missing → cloning only")
        else: print("transformers 4.49 + harness present → skip"); 
except Exception:
    print("transformers wrong/missing → full install")
if need_install:
    if not Path("/content/finetune-hf-vits/run_vits_finetuning.py").exists():
        shell("git clone --depth 1 https://github.com/ylacombe/finetune-hf-vits /content/finetune-hf-vits")
        shell("pip install -q -r /content/finetune-hf-vits/requirements.txt")
        shell("pip install -q 'datasets[audio]>=3.2,<4.0' librosa soundfile accelerate", must=False)
    shell("pip install -q 'transformers==4.49.0'")
    shell("pip install -q 'datasets[audio]>=3.2,<4.0' librosa soundfile accelerate", must=False)
if not glob.glob("/content/finetune-hf-vits/monotonic_align/monotonic_align/core*.so"):
    shell("mkdir -p monotonic_align && python setup.py build_ext --inplace", cwd="/content/finetune-hf-vits/monotonic_align")
shell("mkdir -p /content/finetune-hf-vits/monotonic_align/monotonic_align")
shell("cp -n /content/finetune-hf-vits/monotonic_align/monotonic_align/core.py /content/finetune-hf-vits/monotonic_align/monotonic_align/core.py 2>/dev/null || true")
import transformers, datasets
print("transformers:", transformers.__version__, "| datasets:", datasets.__version__)
assert transformers.__version__.startswith("4."), "wrong transformers — rerun after pin"

# ── S3: HF login (skips if token valid) ──────────────────────────────
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
        raise SystemExit("Login did not complete — run this cell again and finish the login.")

# ── S4: corpus (skips if built) ──────────────────────────────────────
stage(4, "Corpus (mount Drive → unzip → stream audio)")
if not Path("/content/tts_data").exists():
    from google.colab import drive
    drive.mount("/content/drive")
    shell("mkdir -p /content/work && unzip -o -q '/content/drive/MyDrive/UG_TTS/tts_data.zip' -d /content/work")
    base = "https://raw.githubusercontent.com/Dickson32-cell/akan-tts-asante/master/src/"
    for f in ["normalize_twi.py", "extract_asante_colab.py"]:
        Path("/content/" + f).write_text(urllib.request.urlopen(base + f, timeout=60).read().decode("utf-8"), encoding="utf-8")
    r = shell("python /content/extract_asante_colab.py --manifest /content/work/selected.jsonl --out /content/tts_data", must=False)
    if r.returncode != 0:
        print(r.stdout[-800:] if r.stdout else "", r.stderr[-800:] if r.stderr else "")
        raise SystemExit("corpus build failed — send lines above to Hermes")
else:
    print("tts_data present → skip")

# ── S5: base training checkpoint (skips if built) ────────────────────
stage(5, "Training base (generator + Meta discriminator for 'aka')")
if not Path("/content/aka_train_base").exists():
    shell("python /content/finetune-hf-vits/convert_original_discriminator_checkpoint.py --language_code aka --pytorch_dump_folder_path /content/aka_train_base",
          cwd="/content/finetune-hf-vits")
    print("base checkpoint built:", os.listdir("/content/aka_train_base"))
else:
    print("base checkpoint present → skip")

# ── S6: config (always rewritten — gentle settings locked) ───────────
stage(6, "Config (hub id fixed; gentle LR run)")
cfg = {
  "project_name": "akan_twi_tts", "push_to_hub": True,
  "hub_model_id": "Dickson32-cell/akan-twi-mms",
  "report_to": ["tensorboard"], "logging_strategy": "steps", "logging_steps": 20,
  "overwrite_output_dir": False,
  "output_dir": "/content/akan-vits-finetuned",
  "dataset_name": "/content/tts_data", "dataset_config_name": None,
  "audio_column_name": "audio", "text_column_name": "text",
  "train_split_name": "train", "eval_split_name": "eval",
  "speaker_id_column_name": "speaker_id", "filter_on_speaker_id": 0,
  "override_speaker_embeddings": False,
  "full_generation_sample_text": "Mema wo akye, me nuanom. Wo ho te sɛn?",
  "max_duration_in_seconds": 12.0, "min_duration_in_seconds": 2.0, "max_tokens_length": 500,
  "model_name_or_path": "/content/aka_train_base",
  "preprocessing_num_workers": 1,   # 1 avoids fork+BytesIO BufferError (Oct lesson)
  "do_train": True, "num_train_epochs": 20,
  "gradient_accumulation_steps": 1, "per_device_train_batch_size": 16,
  "learning_rate": 1e-4, "lr_scheduler_type": "cosine", "warmup_ratio": 0.02,
  "adam_beta1": 0.8, "adam_beta2": 0.99, "weight_decay": 0.01, "group_by_length": False,
  "do_eval": True, "eval_steps": 500, "per_device_eval_batch_size": 8,
  "max_eval_samples": 25, "do_step_schedule_per_epoch": False,
  "save_steps": 1000, "save_total_limit": 3,
  "weight_disc": 3, "weight_fmaps": 1, "weight_gen": 1, "weight_kl": 1.5,
  "weight_duration": 1, "weight_mel": 35, "fp16": True, "seed": 456
}
Path("/content/finetune_akan.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print("config written (LR 1e-4, 20 epochs, repo Dickson32-cell/akan-twi-mms)")

# ── S7: training — launch fresh / resume / or just monitor ───────────

print("\nSTAGES 1-6 COMPLETE \u2014 continuing to stage 7\u2026")

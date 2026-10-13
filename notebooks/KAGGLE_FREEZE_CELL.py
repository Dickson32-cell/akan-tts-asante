"""KAGGLE PORT of the Akan TTS freeze run (Colab T4 quota exhausted Oct 14).
Single self-contained cell logic; two-file structure preserved via repo raw fetch at run time:
- Kaggle 'Notebook option': create new notebook, enable GPU T4 x2 (Settings) + Internet ON, paste.
Differences vs Colab:
 1. No google.colab.drive — corpus comes from the Drive SHARED folder's direct-download URL
    (public folder, no auth): tts_data.zip (302 KB).
 2. HF login via kagglehub-free interactive getpass token entry.
 3. Output dir /kaggle/working/akan-vits-finetuned-freeze; artifacts saved + pushed same way.
Pins identical: transformers 4.49.0, datasets 3.2.0, seed 456, LR 5e-5, 12 epochs, freeze text_encoder+flow.
"""
# ══════════════ PASTE FROM HERE INTO ONE KAGGLE CELL ══════════════
import subprocess, sys, os, glob, json, time, urllib.request, getpass
from pathlib import Path

def shell(cmd, must=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0 and must:
        print("FAILED:", cmd); print((r.stdout or "")[-900:]); print((r.stderr or "")[-900:])
        raise SystemExit("Send lines above to Hermes")
    return r

print("stage 0: GPU")
r = shell("nvidia-smi --query-gpu=name --format=csv,noheader", must=False)
print("GPU:", (r.stdout or "NONE").strip())
if "T4" not in (r.stdout or "") and "P100" not in (r.stdout or ""):
    SystemExit("No GPU - enable Settings > Accelerator > GPU T4 x2, Internet ON, rerun")
import torch; assert torch.cuda.is_available()
print("cuda ok:", torch.cuda.get_device_name(0))

print("stage 1: pins")
shell("pip install -q 'transformers==4.49.0'")
shell("pip install -q 'datasets==3.2.0' librosa soundfile accelerate")
print("stage 2: harness")
if not Path("/kaggle/tmp/finetune-hf-vits/run_vits_finetuning.py").exists():
    shell("git clone https://github.com/ylacombe/finetune-hf-vits /kaggle/tmp/finetune-hf-vits", must=False)
    shell("pip install -q -r /kaggle/tmp/finetune-hf-vits/requirements.txt", must=False)
    shell("mkdir -p /kaggle/tmp/finetune-hf-vits/monotonic_align/monotonic_align")
    shell("cp -n /kaggle/tmp/finetune-hf-vits/monotonic_align/core.py /kaggle/tmp/finetune-hf-vits/monotonic_align/monotonic_align/core.py || true")
    shell("cd /kaggle/tmp/finetune-hf-vits/monotonic_align && python setup.py build_ext --inplace", must=False)
print("stage 3: HF login")
from huggingface_hub import HfApi
api = HfApi()
try:
    me = api.whoami(); print("logged:", me["name"])
except Exception:
    tok = getpass.getpass("Paste HF token (write): ")
    from huggingface_hub import login; login(tok)
    me = api.whoami(); print("logged:", me["name"])

print("stage 4: corpus from public Drive URL")
if not Path("/kaggle/tmp/tts_data").exists():
    shell("mkdir -p /kaggle/tmp/work")
    shell("curl -L -o /kaggle/tmp/tts_data.zip 'https://drive.google.com/uc?export=download&id=1xM6Z60rNDqgsZ0PREJaMyee' || true")
    r = shell("ls -la /kaggle/tmp/tts_data.zip", must=False)
    size = (r.stdout or "").split()[:0]
    shell("unzip -o -q /kaggle/tmp/tts_data.zip -d /kaggle/tmp/work", must=False)
    shell("cp -r /kaggle/tmp/work/selected.jsonl /kaggle/tmp/work/*  /kaggle/tmp/ 2>/dev/null || true")
    # NOTE: Drive direct download of folders needs confirm token; if zip < 100KB it likely failed -> fallback manual upload
    zpath = "/kaggle/tmp/tts_data.zip"
    if not Path(zpath).exists() or os.path.getsize(zpath) < 100_000:
        raise SystemExit("Drive direct download failed - upload tts_data.zip manually via Kaggle 'Input' (Add Input > Upload), then rerun this cell")
    base = "https://raw.githubusercontent.com/Dickson32-cell/akan-tts-asante/16fedb8/src/"
    for f in ["normalize_twi.py", "extract_asante_colab.py"]:
        Path("/kaggle/tmp/" + f).write_text(urllib.request.urlopen(base+f, timeout=60).read().decode(), encoding="utf-8")
    shell("python /kaggle/tmp/extract_asante_colab.py --manifest /kaggle/tmp/selected.jsonl --out /kaggle/tmp/tts_data", must=False)
print("corpus stage done:", Path("/kaggle/tmp/tts_data").exists())

print("stage 5: base = gentle weights")
from huggingface_hub import snapshot_download
if not glob.glob("/kaggle/tmp/aka_train_base/*.safetensors"):
    snapshot_download(repo_id="Dickson32-cell/akan-twi-mms-gentle", local_dir="/kaggle/tmp/aka_train_base",
                      allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"])
print("base files:", sorted(os.path.basename(x) for x in glob.glob("/kaggle/tmp/aka_train_base/*")))

print("stage 6: config (freeze-run)")
cfg = {
  "project_name": "akan_twi_tts_freeze", "push_to_hub": True,
  "hub_model_id": "Dickson32-cell/akan-twi-mms-freeze",
  "report_to": ["tensorboard"], "logging_strategy": "steps", "logging_steps": 20,
  "overwrite_output_dir": False, "output_dir": "/kaggle/working/akan-vits-finetuned-freeze",
  "dataset_name": "/kaggle/tmp/tts_data", "dataset_config_name": None,
  "audio_column_name": "audio", "text_column_name": "text",
  "train_split_name": "train", "eval_split_name": "eval",
  "speaker_id_column_name": "speaker_id", "filter_on_speaker_id": 0,
  "override_speaker_embeddings": False,
  "full_generation_sample_text": "Mema wo akye, me nuanom. Wo ho te s\u025bn?",
  "max_duration_in_seconds": 12.0, "min_duration_in_seconds": 2.0, "max_tokens_length": 500,
  "model_name_or_path": "/kaggle/tmp/aka_train_base", "preprocessing_num_workers": 1,
  "do_train": True, "num_train_epochs": 12,
  "gradient_accumulation_steps": 1, "per_device_train_batch_size": 16,
  "learning_rate": 5e-5, "lr_scheduler_type": "cosine", "warmup_ratio": 0.02,
  "adam_beta1": 0.8, "adam_beta2": 0.99, "weight_decay": 0.01, "group_by_length": False,
  "do_eval": True, "eval_steps": 500, "per_device_eval_batch_size": 8,
  "max_eval_samples": 25, "do_step_schedule_per_epoch": False,
  "save_steps": 1000, "save_total_limit": 3,
  "weight_disc": 3, "weight_fmaps": 1, "weight_gen": 1, "weight_kl": 1.5,
  "weight_duration": 1, "weight_mel": 35, "fp16": True, "seed": 456
}
resume = bool(glob.glob("/kaggle/working/akan-vits-finetuned-freeze/checkpoint-*"))
if resume: cfg["resume_from_checkpoint"] = "latest"
Path("/kaggle/tmp/finetune_akan_freeze.json").write_text(json.dumps(cfg, indent=2))
print("freeze config written (LR 5e-5, 12ep, hub -freeze)")

print("stage 6b: SAME 4 harness patches + FREEZE insert")
_h = "/kaggle/tmp/finetune-hf-vits/run_vits_finetuning.py"
_s = open(_h, encoding="utf-8").read()
import re
if "load_from_disk(data_args.dataset_name)" not in _s:
    _s = re.sub(r'raw_datasets\[.train.\] = load_dataset\((?:.|\n)*?\)\n',
                'raw_datasets["train"] = load_from_disk(data_args.dataset_name)[data_args.train_split_name]\n', _s, count=1)
    _s = re.sub(r'raw_datasets\[.eval.\] = load_dataset\((?:.|\n)*?\)\n',
                'raw_datasets["eval"] = load_from_disk(data_args.dataset_name)[data_args.eval_split_name]\n', _s, count=1)
    if "from datasets import DatasetDict, load_dataset" in _s:
        _s = _s.replace("from datasets import DatasetDict, load_dataset",
                        "from datasets import DatasetDict, load_dataset, load_from_disk")
    open(_h, "w", encoding="utf-8").write(_s); print("patched: load_from_disk")
_s = open(_h, encoding="utf-8").read()
_old = 'batch[model_input_name] = string_inputs.get("input_ids")[: max_tokens_length + 1]'
if _old in _s:
    _s = _s.replace(_old, 'batch[model_input_name] = string_inputs.get("input_ids")[: int(max_tokens_length) + 1]')
    open(_h, "w", encoding="utf-8").write(_s); print("patched: slice int cast")
_plot = "/kaggle/tmp/finetune-hf-vits/utils/plot.py"
_ps = open(_plot, encoding="utf-8").read()
if "tostring_rgb" in _ps:
    _ps = _ps.replace("data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)",
                      "data = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()")
    open(_plot, "w", encoding="utf-8").write(_ps); print("patched: buffer_rgba")
_vid = sorted(glob.glob("/opt/conda/lib/python3*/site-packages/datasets/features/video.py"))[:1]
if _vid:
    _cur = open(_vid[0], encoding="utf-8", errors="replace").read()
    if not ("except ImportError" in _cur and "VideoReader = None" in _cur):
        _out, _done = [], False
        for _l in _cur.splitlines(keepends=True):
            if not _done and _l.strip() == "from torchvision.io import VideoReader":
                _ind = _l[:len(_l)-len(_l.lstrip())]
                _out += [_ind+"try:", _ind+"    from torchvision.io import VideoReader",
                         _ind+"except ImportError:", _ind+"    VideoReader = None"]
                _done = True
            else: _out.append(_l)
        assert _done
        _new = "".join(_out); compile(_new, _vid[0], "exec")
        open(_vid[0], "w", encoding="utf-8").write(_new + "\n"); print("patched: video.py")
# FREEZE insert after model.eval()
m = re.search(r"\n(\s*)model\.eval\(\)", _s)
assert m
_ins = ("\n# FREEZE-RUN INSERT: protect base phonetics\n_f=0; _k=0\n"
        "for _n, _p in model.named_parameters():\n"
        "    if _n.startswith('text_encoder.') or _n.startswith('flow.'):\n"
        "        _p.requires_grad = False; _f += 1\n"
        "    else:\n        _k += 1\n"
        "print(f'[FREEZE] {_f} frozen / {_k} trainable')\n")
if "# FREEZE-RUN INSERT" not in _s:
    _s = _s[:m.end()] + _ins + _s[m.end():]
    compile(_s, _h, "exec")
    open(_h, "w", encoding="utf-8").write(_s)
    print("patched: FREEZE insert (compile-verified)")

print("stage 7: TRAINING (freeze)")
LOG = "/kaggle/working/train_akan_freeze.log"
cmd = ["python", "/kaggle/tmp/finetune-hf-vits/run_vits_finetuning.py", "/kaggle/tmp/finetune_akan_freeze.json"]
logf = open(LOG, "a")
proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd="/kaggle/tmp/finetune-hf-vits")
print("pid", proc.pid)
last = 0
while proc.poll() is None:
    time.sleep(30)
    try: lines = Path(LOG).read_text(errors="replace").splitlines()
    except FileNotFoundError: continue
    for l in lines[last:]:
        if any(k in l.lower() for k in ("loss","eval_","error","traceback","epoch","it/s","freeze")):
            print(l[:170])
    last = len(lines)
logf.close()
print("=== TRAINING EXITED rc=%s ===" % proc.returncode)
if proc.returncode != 0:
    for l in Path(LOG).read_text(errors="replace").splitlines()[-12:]: print("  ", l[:170])
    raise SystemExit("send lines to Hermes")
print("\nDONE. push to hub = handled by config (hub_model_id -freeze; push_to_hub True)")
print("artifacts: /kaggle/working/akan-vits-finetuned-freeze")
print("NEXT: 4-way eval cell from Hermes after training completes")
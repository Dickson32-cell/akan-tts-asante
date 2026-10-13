configs/ — the exact training launcher configs for ylacombe/finetune-hf-vits.

Reproduce either run with (Colab, after `pip install -r requirements.txt` and
cloning the harness):

  python finetune-hf-vits/run_vits_finetuning.py finetune-hf-vits/configs/akan-twi-gentle.json

IMPORTANT: pass the JSON as the SOLE argument — the harness's argparse mode
takes over if any extra flag is present and the config is then ignored.

- akan-twi-original.json  →  LR 2e-4, 30 epochs = 11,940 steps
  (published at Dickson32-cell/akan-twi-mms; the reviewed research baseline)
- akan-twi-gentle.json    →  LR 1e-4, 20 epochs = 7,960 steps
  (published at Dickson32-cell/akan-twi-mms-gentle; the submitted model)

Both consume the same prepared corpus (./tts_data, built by
src/prepare_dataset.py) and the same seed (456). Colab pins:
transformers==4.49.0, datasets==3.2.0 (see requirements note + scripts/patch_harness_v2.py).
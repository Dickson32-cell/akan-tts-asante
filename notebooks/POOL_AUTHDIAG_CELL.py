# POOL AUTHDIAG - separates kernel-session rights from subprocess env; prints evidence, changes nothing
import os
from huggingface_hub import HfApi
api = HfApi()
w = api.whoami()
acc = (w.get("auth") or {}).get("accessToken") or {}
print("whoami user:", w.get("name"))
print("token type:", acc.get("type"), "| role:", acc.get("role"))
print("stored token file:", os.path.exists("/root/.cache/huggingface/token"))
print("env HF_TOKEN present:", bool(os.environ.get("HF_TOKEN")), "| HUGGING_FACE_HUB_TOKEN present:", bool(os.environ.get("HUGGING_FACE_HUB_TOKEN")))
try:
    r = api.create_repo("Dickson32-cell/akan-twi-mms-pool", repo_type="model", exist_ok=True)
    print("kernel-side create_repo: OK ->", r)
except Exception as e:
    print("kernel-side create_repo FAILED:", str(e)[:200])

# The POOL-RUN loader - the only cell to run in Colab (T4).
# Fetches the two pool-run stage files from GitHub at run time (pinned) and executes them.
# Stages auto-skip what is already done; checkpoints mirror to Drive every save; resume-on-contact.
import urllib.request

BASE = "https://raw.githubusercontent.com/Dickson32-cell/akan-tts-asante/edbf10c/notebooks/"
def pull(name):
    code = urllib.request.urlopen(BASE + name, timeout=60).read().decode("utf-8")
    print("pulled", name, f"({len(code)} bytes)")
    return code

print("=== AKAN TTS POOL-RUN LOADER (P1: 39.98h full pool) ===")
print("run: base=facebook/mms-tts-aka | LR 1e-4 | 15 epochs | seed 456 | checkpoints mirrored to Drive | repo akan-twi-mms-pool")
print("-" * 70)
part1 = pull("pool_stage1.py")
part2 = pull("pool_stage7.py")

print("\n--- part 1: stages 1-5 (GPU/harness/login/corpus 40h/base checkpoint) ---")
try:
    exec(compile(part1, "pool_stage1", "exec"), globals())
except SystemExit as e:
    print("STOPPED:", e)
    print("(fix what it named, then run this cell again - it continues where it stopped)")
    raise
except Exception:
    import traceback; traceback.print_exc()
    print("The lines above state the failing stage and its fix.")
    raise

print("\n--- part 2: stages 6-9 (patches -> pool training + Drive mirror -> convergence -> push) ---")
try:
    exec(compile(part2, "pool_stage7", "exec"), globals())
except SystemExit as e:
    print("STOPPED:", e)
    print("(checkpoints ARE mirrored in Drive - rerunning this cell RESUMES from the newest)")
    raise
except Exception:
    import traceback; traceback.print_exc()
    print("The lines above state the failing stage and its fix.")
    raise

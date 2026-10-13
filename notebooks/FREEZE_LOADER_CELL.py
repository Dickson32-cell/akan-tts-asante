# ═══ THE FREEZE-RUN LOADER — the only cell to run in Colab ═══
# Fetches the two freeze-run stage files from GitHub at run time (pinned) and executes them.
# Stages auto-skip what's already done. Put this whole cell in a FRESH Colab notebook (T4 GPU) and run it.
import urllib.request

BASE = "https://raw.githubusercontent.com/Dickson32-cell/akan-tts-asante/6162cd7/notebooks/"
def pull(name):
    code = urllib.request.urlopen(BASE + name, timeout=60).read().decode("utf-8")
    print("pulled", name, f"({len(code)} bytes)")
    return code

print("═══ AKAN TTS FREEZE-RUN LOADER ═══")
print("run: base=gentle weights | freeze text_encoder+flow | LR 5e-5 | NEW repo -freeze | main+gentle untouchable")
print("-" * 70)
part1 = pull("freeze_stage1.py")
part2 = pull("freeze_stage7.py")

print("\n──── part 1: stages 1-5 (GPU/harness/login/corpus/freeze-base) ────")
try:
    exec(compile(part1, "freeze_stage1", "exec"), globals())
except SystemExit as e:
    print("STOPPED:", e)
    print("(fix what it named, then run this cell again - it continues where it stopped)")
    raise
except Exception:
    import traceback; traceback.print_exc()
    print("The lines above state the failing stage and its fix.")
    raise

print("\n──── part 2: stages 6-9 (patches -> FREEZE training -> convergence -> isolated push) ────")
try:
    exec(compile(part2, "freeze_stage7", "exec"), globals())
except SystemExit as e:
    print("STOPPED:", e)
    print("(training failures: review the printed lines on failure)")
    raise
except Exception:
    import traceback; traceback.print_exc()
    print("The lines above state the failing stage and its fix.")
    raise
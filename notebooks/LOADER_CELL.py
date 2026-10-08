# ═══ THE LOADER — the only cell to run in Colab ═══
# 1: stages 1-6 (GPU, harness, login, data, base)  2: stages 7-9 (train→report→push)
# If ANY stage was already done in this session, it skips it automatically.
# Copy this ENTIRE cell into a fresh Colab cell, press play, follow its prompts.
import urllib.request, sys

BASE = "https://raw.githubusercontent.com/Dickson32-cell/akan-tts-asante/de99c94/notebooks/"
def pull(name):
    url = BASE + name
    code = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")
    print("pulled", name, f"({len(code)} bytes)")
    return code

print("═══ AKAN TTS LOADER ═══")
part1 = pull("master_stage1.py")
part2 = pull("master_stage7.py")

print("\n──── executing part 1 (stages 1-6) ────")
try:
    exec(compile(part1, "stage1-6", "exec"), globals())
except SystemExit as e:
    print("STOPPED:", e)
    print("(fix whatever it named, then run this cell again — it continues where it stopped)")
    raise
except Exception:
    import traceback; traceback.print_exc()
    print("Send the lines above to Hermes in Telegram.")
    raise

print("\n──── executing part 2 (stages 7-9: training → convergence → push) ────")
try:
    exec(compile(part2, "stage7-9", "exec"), globals())
except SystemExit as e:
    print("STOPPED:", e)
    print("(training failed or a stage refused — send Hermes the lines above)")
    raise SystemExit(0)
except Exception:
    import traceback; traceback.print_exc()
    print("Send the lines above to Hermes in Telegram.")
    raise SystemExit(0)
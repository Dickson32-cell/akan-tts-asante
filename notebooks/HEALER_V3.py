# Akan TTS — HEALER v3 (compile-verified, span-rebuild, self-diagnosing)
# Purpose: repair datasets/features/video.py wounded by an earlier bad patch.
# Run: paste ONE two-liner in Colab that pulls this file and execs it.
import subprocess, glob, os

def find_video_dot_py():
    pats = [
        "/usr/local/lib/python3*/dist-packages/datasets/features/video.py",
        "/usr/local/lib/python3*/site-packages/datasets/features/video.py",
        "/usr/lib/python3*/dist-packages/datasets/features/video.py",
    ]
    hits = []
    for p in pats:
        hits += glob.glob(p)
    hits = [h for h in hits if os.path.exists(h)]
    if not hits:
        import datasets
        hits = [os.path.join(os.path.dirname(datasets.__file__), "features", "video.py")]
    return hits[0]

vid = find_video_dot_py()
print("FILE:", vid)
src = open(vid, encoding="utf-8", errors="replace").read()

def ok(txt):
    try:
        compile(txt, vid, "exec"); return True
    except Exception:
        return False

if ok(src) and "from torchvision.io import VideoReader" in src:
    print("STATE: compiles + import present -> nothing to heal")
    sys_exit_ok = True
else:
    print("STATE: file does NOT compile — rebuilding the damaged region")
    lines = src.splitlines(keepends=True)
    print("--- first 30 lines (repr) ---")
    for i, l in enumerate(lines[:30], 1):
        print(f"{i:3d}| {l.rstrip()[:110]!r}")

    # strategy: find every line whose content mentions VideoReader-import; rebuild the region
    imp_idx = [i for i, l in enumerate(lines) if "torchvision.io import VideoReader" in l]
    print("VideoReader-import lines at:", imp_idx)
    # remove ALL lines from the FIRST 'try:'-like line preceding the first import
    # through the LAST line of the last import-region block
    first_imp = imp_idx[0]
    # walk back from first_imp up to 8 lines to find a 'try' or damaged scaffold start
    start = first_imp
    for j in range(first_imp, max(0, first_imp - 8), -1):
        if lines[j].strip() in ("try:", "try:  # optional import"):
            start = j; break
    # walk forward up to 8 lines after the LAST import line to find the except/None scaffold end
    last_imp = imp_idx[-1]
    end = last_imp
    for j in range(last_imp, min(last_imp + 8, len(lines))):
        if "VideoReader = None".lower() in lines[j].lower():
            end = j; break
    indent = ""
    canon = [
        indent + "try:\n",
        indent + "    from torchvision.io import VideoReader\n",
        indent + "except ImportError:\n".replace("on", "except"),   # literal 'except ImportError:'
        indent + "    VideoReader = None\n",
    ]
    out = lines[:start] + canon + lines[end + 1:]
    new = "".join(out)
    if not ok(new):
        # last resort: fully regenerate a canonical header — remove the import region entirely and append guarded import at a safe spot: simplest = wrap in function-free guarded import at the very top after docstring; easiest compile-safe version:
        without = "".join(lines[:start] + lines[end + 1:])
        tail_block = (
            "\n# PATCH (env drift): guarded optional import (torchvision 0.20+ removed VideoReader)\n"
            "try:\n"
            "    from torchvision.io import VideoReader\n"
            "except ImportError:\n"
            "    VideoReader = None\n"
        )
        new = without.rstrip() + tail_block
    assert ok(new), "FINAL REBUILD STILL FAILS — full dump follows"
    open(vid, "w", encoding="entry" if False else "utf-8").write(new)
    print("REBUILT + compile-verified")
    print("--- first 30 lines after ---")
    for i, l in enumerate(open(vid, encoding="utf-8").read().splitlines()[:30], 1):
        print(f"{i:3d}| {l.rstrip()[:110]!r}")

# fresh-subprocess probe (the real test: a virgin python must import datasets fine)
r = subprocess.run(["python", "-c", "import datasets; print('REIMPORT-OK', datasets.__version__)"],
                   capture_output=True, text=True)
tail = (r.stdout or r.stderr).strip().splitlines()
print("PROBE [rc=%d]:" % r.returncode, tail[-1][:110] if tail else "(none)")
if r.returncode != 0:
    print("--- final 22 lines of the file ---")
    for i, l in enumerate(open(vid, encoding="utf-8").read().splitlines()[:22], 1):
        print(f"{i:3d}| {l.rstrip()[:110]!r}")
    raise SystemExit(0)
print("READY - now run the loader cell (BASE pin 8b9cc0e) to start training.")

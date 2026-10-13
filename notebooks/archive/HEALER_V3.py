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
    print("STATE: file does NOT compile — rebuilding the TYPE_CHECKING region (heal v4)")
    lines = src.splitlines(keepends=True)
    print("--- first 30 lines (repr) ---")
    for i, l in enumerate(lines[:30], 1):
        print(f"{i:3d}| {l.rstrip()[:110]!r}")
    ti = next(i for i, l in enumerate(lines) if l.strip() == "if TYPE_CHECKING:")
    fi = next(i for i in range(ti, min(ti + 12, len(lines)))
              if "from .features import FeatureType" in lines[i])
    canon = [
        "if TYPE_CHECKING:\n",
        "    try:\n",
        "        from torchvision.io import VideoReader\n",
        "    except ImportError:\n",
        "        VideoReader = None\n",
        "\n",
        "    from .features import FeatureType\n",
    ]
    out = lines[:ti] + canon + lines[fi + 1:]
    new = "".join(out)
    assert ok(new), "FINAL REBUILD STILL FAILS — full dump follows"
    open(vid, "w", encoding="utf-8").write(new)
    print("REBUILT + compile-verified (TYPE_CHECKING span regenerated)")
    print("--- first 26 lines after ---")
    for i, l in enumerate(open(vid, encoding="utf-8").read().splitlines()[:26], 1):
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

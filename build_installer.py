"""
Builds the installer AND the update for the previous release:

  dist\\MeetingIsland-Setup.exe             first-time install (everything: program + models + runtime), ~410 MB
  dist\\MeetingIsland-Update-to-<ver>.exe   only the files that changed since the release it was built against (usually a few
                                           MB), applies itself to the copy that is already installed

  1. PyInstaller turns the Python app into MeetingIsland.exe + libraries (skipped with --no-freeze)
  2. the speech models, the Visual C++ runtime, the guide and the uninstaller are added
  3. a manifest (size + sha256 of every file) of this release is stored in releases\\<ver>\\
  4. the full installer is zipped and wrapped by Windows' own IExpress; the update is the difference to an earlier manifest

Run from this folder:
  .venv\\Scripts\\python.exe build_installer.py                  full installer + update from the latest earlier release
  .venv\\Scripts\\python.exe build_installer.py --update-only    just the (fast) update package
  .venv\\Scripts\\python.exe build_installer.py --no-freeze      reuse dist\\MeetingIsland from the last PyInstaller run
  .venv\\Scripts\\python.exe build_installer.py --update-from 1.0  pick the release the update is built against
The version number comes from version.txt: raise it before every release you hand out.
Only Windows built-ins are used for packaging (IExpress, tar), nothing is downloaded.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIST = HERE / "dist"
APP = DIST / "MeetingIsland"
STAGE = HERE / "build" / "stage"
PAYLOAD = HERE / "build" / "payload"
UPDATE = HERE / "build" / "update_payload"
RELEASES = HERE / "releases"
SETUP = DIST / "MeetingIsland-Setup.exe"
VERSION = (HERE / "version.txt").read_text(encoding="ascii").strip()

# the models the app really loads (see live.py / engine.py / speakers.py / polish.py); the rest of models/ is experiments
MODELS = {
    "hinglish-swift-ct2": None,                                  # whole folder
    "indicconformer-hi": None,
    "speaker": ["nemo_en_titanet_small.onnx"],                   # only these files
}
EXTRA_MODEL_DIRS = {"speaker/sherpa-onnx-pyannote-segmentation-3-0": ["model.int8.onnx"]}     # who-speaks-when (second pass)
VC_DLLS = ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
           "concrt140.dll", "vcomp140.dll"]                      # so a laptop without the VC++ redistributable still runs


def run(cmd, **kw):
    print(">", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def freeze():
    shutil.rmtree(HERE / "build" / "MeetingIsland", ignore_errors=True)
    shutil.rmtree(APP, ignore_errors=True)
    env = dict(os.environ, HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--distpath", str(DIST),
         "--workpath", str(HERE / "build"), str(HERE / "MeetingIsland.spec")], env=env)


def stage():
    shutil.rmtree(STAGE, ignore_errors=True)
    shutil.copytree(APP, STAGE, ignore=shutil.ignore_patterns("models"))          # (dev builds link models in; do not follow)
    internal = STAGE / "_internal"
    for dll in VC_DLLS:
        src = Path(os.environ["SystemRoot"]) / "System32" / dll
        if src.exists() and not (internal / dll).exists():
            shutil.copy2(src, internal / dll)
            print("added", dll)
    for name, only in MODELS.items():
        src, dst = HERE / "models" / name, STAGE / "models" / name
        if not src.exists():
            sys.exit(f"missing model: {src}")
        if only is None:
            shutil.copytree(src, dst)
        else:
            dst.mkdir(parents=True, exist_ok=True)
            for f in only:
                shutil.copy2(src / f, dst / f)
    for name, only in EXTRA_MODEL_DIRS.items():
        src, dst = HERE / "models" / name, STAGE / "models" / name
        if not src.exists():
            sys.exit(f"missing model: {src}")
        dst.mkdir(parents=True, exist_ok=True)
        for f in only:
            shutil.copy2(src / f, dst / f)
    inst = HERE / "installer"
    shutil.copy2(HERE / "island.ico", STAGE / "island.ico")
    shutil.copy2(inst / "QUICK-START.txt", STAGE / "QUICK-START.txt")
    shutil.copy2(inst / "uninstall.ps1", STAGE / "uninstall.ps1")
    shutil.copy2(inst / "Uninstall Meeting Island.cmd", STAGE / "Uninstall Meeting Island.cmd")
    (STAGE / "VERSION.txt").write_text(f"{VERSION}\nbuilt {time.strftime('%Y-%m-%d %H:%M')}\n", encoding="ascii")
    total = sum(f.stat().st_size for f in STAGE.rglob("*") if f.is_file())
    print(f"staged {total / 1e6:.0f} MB")


# ------------------------------------------------------------------------------------------------ manifests
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_manifest():
    """size + sha256 of every staged file -> releases/<ver>/manifest.json, and manifest.tsv inside the app (read by the updater)."""
    files = {}
    for f in sorted(STAGE.rglob("*")):
        if f.is_file() and f.name != "manifest.tsv":
            files[f.relative_to(STAGE).as_posix()] = dict(size=f.stat().st_size, sha256=sha256(f))
    out = RELEASES / VERSION
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(dict(version=VERSION, files=files)), encoding="utf-8")
    tsv = "\n".join(f"{rel}\t{v['size']}\t{v['sha256']}" for rel, v in files.items()) + "\n"
    (STAGE / "manifest.tsv").write_text(tsv, encoding="utf-8", newline="\n")
    print(f"manifest for {VERSION}: {len(files)} files")
    return files


def previous_release():
    vs = []
    for d in RELEASES.iterdir() if RELEASES.exists() else []:
        if (d / "manifest.json").exists() and d.name != VERSION:
            vs.append(d.name)
    key = lambda v: [int(x) if x.isdigit() else 0 for x in v.split(".")]
    vs = [v for v in vs if key(v) < key(VERSION)]
    return max(vs, key=key) if vs else None


# ------------------------------------------------------------------------------------------------ IExpress
def iexpress(target, src_dir, files, friendly, launch):
    """Wrap the files in src_dir into one self-extracting exe that runs `launch` (one of the files) after unpacking."""
    Path(target).unlink(missing_ok=True)
    sed = HERE / "build" / (Path(target).stem + ".sed")
    lines = [
        "[Version]", "Class=IEXPRESS", "SEDVersion=3",
        "[Options]", "PackagePurpose=InstallApp", "ShowInstallProgramWindow=0", "HideExtractAnimation=1",
        "UseLongFileName=1", "InsideCompressed=0", "CAB_FixedSize=0", "CAB_ResvCodeSigning=0", "RebootMode=N",
        "InstallPrompt=", "DisplayLicense=", "FinishMessage=",
        f"TargetName={target}", f"FriendlyName={friendly}",
        # the bare file name: IExpress silently launches nothing when the command starts with "cmd /c"
        f"AppLaunched={launch}", "PostInstallCmd=<None>", f"AdminQuietInstCmd={launch}", f"UserQuietInstCmd={launch}",
        "SourceFiles=SourceFiles", f"FILE_COUNT={len(files)}",
        "[Strings]",
    ]
    lines += [f'FILE{i}="{n}"' for i, n in enumerate(files)]
    lines += ["[SourceFiles]", f"SourceFiles0={src_dir}\\", "[SourceFiles0]"]
    lines += [f"%FILE{i}%=" for i in range(len(files))]
    sed.write_text("\r\n".join(lines) + "\r\n", encoding="ascii")
    t = time.time()
    run([str(Path(os.environ["SystemRoot"]) / "System32" / "iexpress.exe"), "/N", "/Q", str(sed)])
    for _ in range(600):                                   # iexpress returns before the file is completely written
        if Path(target).exists():
            a = Path(target).stat().st_size
            time.sleep(2)
            if Path(target).stat().st_size == a:
                break
        time.sleep(1)
    if not Path(target).exists():
        sys.exit("IExpress did not produce " + str(target))
    print(f"{target}  {Path(target).stat().st_size / 1e6:.1f} MB  ({time.time() - t:.0f}s)")


# ------------------------------------------------------------------------------------------------ full installer
def make_zip():
    shutil.rmtree(PAYLOAD, ignore_errors=True)
    PAYLOAD.mkdir(parents=True)
    zpath = PAYLOAD / "app.zip"
    t = time.time()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=4, allowZip64=True) as z:
        for f in sorted(STAGE.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(STAGE).as_posix())
    print(f"app.zip {zpath.stat().st_size / 1e6:.0f} MB in {time.time() - t:.0f}s")
    shutil.copy2(HERE / "installer" / "install.cmd", PAYLOAD / "install.cmd")
    shutil.copy2(HERE / "installer" / "install.ps1", PAYLOAD / "install.ps1")


def make_setup():
    iexpress(SETUP, PAYLOAD, ["install.cmd", "app.zip", "install.ps1"], "Meeting Island Setup", "install.cmd")


# ------------------------------------------------------------------------------------------------ update
def make_update(new_files, base_version):
    base = json.loads((RELEASES / base_version / "manifest.json").read_text(encoding="utf-8"))["files"]
    changed = [r for r, v in new_files.items() if r not in base or base[r]["sha256"] != v["sha256"]]
    removed = [r for r in base if r not in new_files]
    size = sum(new_files[r]["size"] for r in changed)
    print(f"update {base_version} -> {VERSION}: {len(changed)} changed/new files ({size / 1e6:.1f} MB), {len(removed)} removed")
    shutil.rmtree(UPDATE, ignore_errors=True)
    UPDATE.mkdir(parents=True)
    with zipfile.ZipFile(UPDATE / "update.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for r in changed:
            z.write(STAGE / r, r)
    (UPDATE / "changed.txt").write_text("\n".join(changed) + "\n", encoding="utf-8")
    (UPDATE / "removed.txt").write_text("\n".join(removed) + "\n", encoding="utf-8")
    (UPDATE / "plan.txt").write_text(f"from={base_version}\nto={VERSION}\n", encoding="ascii")
    shutil.copy2(STAGE / "manifest.tsv", UPDATE / "manifest.tsv")
    shutil.copy2(HERE / "installer" / "update.cmd", UPDATE / "update.cmd")
    shutil.copy2(HERE / "installer" / "update.ps1", UPDATE / "update.ps1")
    out = DIST / f"MeetingIsland-Update-to-{VERSION}.exe"
    iexpress(out, UPDATE, ["update.cmd", "update.zip", "manifest.tsv", "changed.txt", "removed.txt", "plan.txt", "update.ps1"],
             f"Meeting Island Update {VERSION}", "update.cmd")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-freeze", action="store_true", help="reuse dist\\MeetingIsland from the last PyInstaller run")
    ap.add_argument("--update-only", action="store_true", help="skip the full installer (it takes ~10 minutes)")
    ap.add_argument("--update-from", default=None, help="release the update is built against (default: the latest earlier one)")
    a = ap.parse_args()
    print("version", VERSION)
    if not a.no_freeze:
        freeze()
    stage()
    files = make_manifest()
    base = a.update_from or previous_release()
    if base:
        make_update(files, base)
    else:
        print("no earlier release in releases\\ - no update package built")
    if not a.update_only:
        make_zip()
        make_setup()
        shutil.copy2(SETUP, RELEASES / VERSION / f"MeetingIsland-Setup-{VERSION}.exe")

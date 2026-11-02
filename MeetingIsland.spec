# PyInstaller recipe for Meeting Island. Build with:  .venv\Scripts\python.exe build_installer.py
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ("wordfreq", "langcodes", "language_data", "onnx_asr", "sherpa_onnx", "pyaudiowpatch", "ctranslate2",
            "faster_whisper", "av", "onnxruntime", "tokenizers", "mss", "metaphone", "regex", "ftfy", "msgpack",
            "dxcam", "comtypes", "winrt", "winrt.runtime", "winrt.windows.media.ocr", "winrt.windows.graphics.imaging",
            "winrt.windows.storage.streams", "winrt.windows.globalization", "winrt.windows.foundation",
            "winrt.windows.foundation.collections"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:                       # an optional helper that is not installed is fine
        print("collect_all skipped", pkg, e)

hiddenimports += ["engine", "fusion", "speakers", "glossary", "translit", "paths", "backend", "live", "assistant",
                  "recorder", "polish", "finisher", "screennames", "corrections"]

a = Analysis(
    ["island_qt.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "scipy", "IPython", "pytest", "PIL", "imageio_ffmpeg", "pandas", "torch",
              "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQml", "PySide6.QtQuick",
              "PySide6.Qt3DCore", "PySide6.QtCharts", "PySide6.QtMultimedia", "PySide6.QtPdf", "PySide6.QtSql"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MeetingIsland",
    console=False,                # no black window: the app logs to %LOCALAPPDATA%\MeetingIsland\island.log
    icon="island.ico",
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="MeetingIsland")

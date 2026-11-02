"""
Downloads the four models Meeting Island needs into models/ (about 330 MB; none of them are stored in this repository).

    python tools/fetch_models.py

  hinglish-swift-ct2        Oriserve/Whisper-Hindi2Hinglish-Swift (Apache-2.0), a Whisper-base fine-tune that writes Hinglish in Roman
                            script. Converted here to CTranslate2 int8 (needs: pip install torch transformers ctranslate2).
  indicconformer-hi         AI4Bharat IndicConformer, Hindi, int8 ONNX, as packaged for onnx-asr (OpenVoiceOS/ai4bharat-indicconformer-hi-onnx)
  speaker/nemo_en_titanet_small.onnx                      NVIDIA TitaNet-small voice prints (sherpa-onnx model release)
  speaker/sherpa-onnx-pyannote-segmentation-3-0/          pyannote segmentation 3.0, ONNX (who speaks when)

Check each model's own licence before redistributing anything built from it.
"""
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"


def get(url, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    print("downloading", url, flush=True)
    urllib.request.urlretrieve(url, dst)


def swift():
    out = MODELS / "hinglish-swift-ct2"
    if (out / "model.bin").exists():
        return print("hinglish-swift-ct2 already there")
    cmd = [sys.executable, "-m", "ctranslate2.converters.transformers", "--model", "Oriserve/Whisper-Hindi2Hinglish-Swift",
           "--output_dir", str(out), "--quantization", "int8", "--copy_files", "tokenizer.json", "preprocessor_config.json", "--force"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def indic():
    out = MODELS / "indicconformer-hi"
    if (out / "model.int8.onnx").exists():
        return print("indicconformer-hi already there")
    from huggingface_hub import snapshot_download
    snapshot_download("OpenVoiceOS/ai4bharat-indicconformer-hi-onnx", local_dir=str(out),
                      allow_patterns=["config.json", "model.int8.onnx", "vocab.txt"])


def speaker():
    d = MODELS / "speaker"
    if not (d / "nemo_en_titanet_small.onnx").exists():
        get(f"{SHERPA}/speaker-recongition-models/nemo_en_titanet_small.onnx", d / "nemo_en_titanet_small.onnx")
    if not (d / "sherpa-onnx-pyannote-segmentation-3-0" / "model.int8.onnx").exists():
        tmp = d / "seg.tar.bz2"
        get(f"{SHERPA}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2", tmp)
        with tarfile.open(tmp) as t:
            t.extractall(d)
        tmp.unlink()


if __name__ == "__main__":
    for step in (indic, speaker, swift):
        step()
    print("models ready in", MODELS)

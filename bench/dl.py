import os, time, sys
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'; os.environ['HF_HUB_DISABLE_XET'] = '1'; os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
from huggingface_hub import snapshot_download
jobs = [('songzewu/vasista22-whisper-hindi-small-ct2', 'models/hindi-small-ct2', ['model.bin', 'config.json', 'tokenizer.json', 'vocabulary.json', 'preprocessor_config.json', 'vocabulary.txt']),
        ('collabora/faster-whisper-medium-hindi', 'models/hindi-medium-ct2', None)]
for repo, dst, allow in jobs:
    for attempt in range(8):
        try:
            p = snapshot_download(repo, local_dir=dst, allow_patterns=allow); print('OK', repo, os.listdir(dst), flush=True); break
        except Exception as e:
            print('retry', repo, type(e).__name__, str(e)[:100], flush=True); time.sleep(8)
print('DL_DONE', flush=True)

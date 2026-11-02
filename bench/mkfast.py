"""Make faster-speech versions of the benchmark clips (pitch preserved) with the same reference transcripts."""
import json, subprocess, sys, os
import imageio_ffmpeg
ff = imageio_ffmpeg.get_ffmpeg_exe()
meta = json.load(open('bench/meta.json', encoding='utf-8'))
for factor in (1.3, 1.6):
    out_dir = f'bench/data_x{factor}'; os.makedirs(out_dir, exist_ok=True)
    new = []
    for m in meta:
        dst = f"{out_dir}/{os.path.basename(m['file'])}.wav"
        if not os.path.exists(dst):
            subprocess.run([ff, '-y', '-loglevel', 'error', '-i', m['file'], '-filter:a', f'atempo={factor}', '-ar', '16000', '-ac', '1', dst], check=True)
        new.append(dict(m, file=dst, id=m['id'] + f'_x{factor}'))
    json.dump(new, open(f'bench/meta_x{factor}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(factor, len(new), 'clips')

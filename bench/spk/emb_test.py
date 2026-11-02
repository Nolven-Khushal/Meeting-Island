import json, sys, time, itertools, random
import numpy as np
sys.path.insert(0, '.')
import sherpa_onnx, assistant
SR = 16000
meta = json.load(open('bench/spk/meta.json'))
audio = [assistant.decode_audio(m['file']).astype(np.float32) for m in meta]
MODELS = {'wespeaker_resnet34_LM': 'models/speaker/wespeaker_en_voxceleb_resnet34_LM.onnx',
          'campplus_en': 'models/speaker/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx',
          'titanet_small': 'models/speaker/nemo_en_titanet_small.onnx'}
def embed_all(path):
    ex = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=path, num_threads=2, debug=False, provider='cpu'))
    t = time.perf_counter(); out = []
    for a in audio:
        s = ex.create_stream(); s.accept_waveform(sample_rate=SR, waveform=a); s.input_finished(); out.append(np.array(ex.compute(s), np.float32))
    return np.stack(out), (time.perf_counter() - t) / sum(len(a) for a in audio) * SR
labels = np.array([m['spk'] for m in meta])
for name, path in MODELS.items():
    E, rtf = embed_all(path); E /= np.linalg.norm(E, axis=1, keepdims=True)
    S = E @ E.T; iu = np.triu_indices(len(E), 1); same = (labels[:, None] == labels[None, :])[iu]; sc = S[iu]
    # best single threshold + equal error rate
    ths = np.linspace(sc.min(), sc.max(), 400); best = max(((np.mean(np.where(same, sc >= t, sc < t)), t) for t in ths))
    far = np.array([np.mean(sc[~same] >= t) for t in ths]); frr = np.array([np.mean(sc[same] < t) for t in ths]); eer = ths[np.argmin(abs(far - frr))]
    print(f"{name:22s} dim {E.shape[1]:3d} | embed time {rtf:.3f}x real time | same-speaker sim {sc[same].mean():.2f} vs different {sc[~same].mean():.2f} | "
          f"best pair accuracy {100*best[0]:.1f}% at threshold {best[1]:.2f} | EER {100*np.mean(np.where(same, sc < eer, sc >= eer)):.1f}%")
    np.save(f'bench/spk/emb_{name}.npy', E)

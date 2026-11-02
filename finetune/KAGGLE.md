# Fine-tuning the Hinglish model (free GPU, a few hours)

The app's English/Hinglish speech model (Oriserve's Whisper-Hindi2Hinglish-Swift, a Whisper-base fine-tune, 72 M parameters) can be
trained further on more Hinglish speech. `train_swift.py` does it on a free Kaggle or Colab GPU and exports the result in the format
the app uses (CTranslate2, int8, about 76 MB). `eval_candidate.py` decides, on this PC, whether the new model may replace the old one.

## On Kaggle
1. Free account at kaggle.com, phone-verified (unlocks the GPU). **Create > New Notebook**, Accelerator **GPU T4**, Internet **On**.
2. Upload `train_swift.py` and `translit.py` from this folder to the notebook.
3. Run:
```python
!pip -q install ctranslate2 soundfile datasets scipy
!python train_swift.py --smoke        # 3 minutes: checks that everything works, ends with "DONE"
!python train_swift.py --hours 150    # the real run, about 1.5 to 3 hours
```
   It streams a slice of the public dataset `agarwalayushi/hinglish` (CC BY 4.0; 2,264 h of Hindi, Hinglish and Indian English),
   turns the transcripts into Roman Hinglish with `translit.py`, adds noise / background voices / speed changes, learns gently
   (learning rate 1e-5), prints the word error rate on clips it never trained on, and writes `swift_ft/swift_ft_ct2.zip`.
4. Download the zip and unzip it to a folder, e.g. `finetune/swift_ft_ct2`.

## Decide on your own machine
```
python finetune/eval_candidate.py finetune/swift_ft_ct2
```
Old and new model run on the same clips: the 80 benchmark clips in `bench/` (lecture and spontaneous conversation, with human
transcripts) and, if you provide one, a hard TV-dialogue clip. It prints `SHIP IT` only if all of these hold: overall error at least
1.0 point lower, conversation not worse (+0.5), lecture not worse (+1.0), the TV clip not worse, punctuation kept (>= 80%).
A model trained mostly on clean lectures and read speech can get worse on noisy real conversation; this check is what prevents it
from shipping.

## If the first result is not good enough
- Worse on conversation: `--noise 0.7 --lr 5e-6` (more noise, gentler) or fewer hours.
- Not better on lecture speech: `--hours 300`.
- The largest gain is expected from in-domain data: add your own recorded meetings and corrected transcripts as extra training clips.

# emojiFilter

Real-time webcam filter: face detection, 8-class emotion classification, emoji overlay.

![Demo](assets/demo.jpg)

Stack: Python, OpenCV, TensorFlow/Keras.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Requires Python 3.9, a webcam, and `assets/models/emotions.h5` (not included — train via `notebooks/emotions.ipynb` or copy weights into `assets/models/`).

Press `q` to quit.

## Training

```bash
pip install -r requirements-train.txt
jupyter notebook notebooks/emotions.ipynb
```

Organize labeled face images under `data/`:

```
data/
├── train/
│   ├── anger/
│   ├── contempt/
│   ├── disgust/
│   ├── fear/
│   ├── happiness/
│   ├── neutral/
│   ├── sadness/
│   └── surprise/
└── test/          # optional, same layout
```

Folder names must match `emoji_filter.config.EMOTIONS`. Exports to `assets/models/emotions.h5`.

## License

MIT — see [LICENSE](LICENSE).

Face detection uses [OpenCV's Haar cascade](https://github.com/opencv/opencv/tree/master/data/haarcascades) (`assets/haarcascade_frontalface_default.xml`).

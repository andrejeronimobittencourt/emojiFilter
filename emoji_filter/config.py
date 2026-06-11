"""Project paths and runtime constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / 'assets'
MODEL_PATH = ASSETS_DIR / 'models' / 'emotions.h5'
EMOJI_DIR = ASSETS_DIR / 'emoji'
FACE_CASCADE_PATH = ASSETS_DIR / 'haarcascade_frontalface_default.xml'

EMOTIONS = [
  'anger',
  'contempt',
  'disgust',
  'fear',
  'happiness',
  'neutral',
  'sadness',
  'surprise',
]

INPUT_SIZE = (150, 150)
DEFAULT_CAMERA_INDEX = 0
QUIT_KEY = 'q'

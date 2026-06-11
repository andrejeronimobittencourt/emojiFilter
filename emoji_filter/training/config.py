"""Paths and hyperparameters for model training."""

from pathlib import Path

from emoji_filter.config import EMOTIONS, INPUT_SIZE, PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / 'data'
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR = DATA_DIR / 'test'
MODEL_EXPORT_PATH = PROJECT_ROOT / 'assets' / 'models' / 'emotions.h5'

BATCH_SIZE = 100
IMAGE_SIZE = INPUT_SIZE[0]
VALIDATION_SPLIT = 0.2
EPOCHS_CNN = 55
EPOCHS_TRANSFER = 10

# Folder names must match EMOTIONS so exported models align with runtime emoji assets.
REQUIRED_CLASS_DIRS = EMOTIONS

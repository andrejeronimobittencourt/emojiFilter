"""Training data loading and Keras image generators."""

from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from emoji_filter.training.config import (
  BATCH_SIZE,
  IMAGE_SIZE,
  REQUIRED_CLASS_DIRS,
  TEST_DIR,
  TRAIN_DIR,
  VALIDATION_SPLIT,
)


def summarize_data(data_dir: Path) -> dict[str, int]:
  """Count images per class folder."""
  counts: dict[str, int] = {}
  missing: list[str] = []

  for class_name in REQUIRED_CLASS_DIRS:
    class_dir = data_dir / class_name
    if not class_dir.is_dir():
      missing.append(class_name)
      continue
    counts[class_name] = len(list(class_dir.iterdir()))

  if missing:
    missing_list = ', '.join(missing)
    raise FileNotFoundError(
      f'Missing class folders under {data_dir}: {missing_list}. '
      'Expected folder names from emoji_filter.config.EMOTIONS.',
    )

  return counts


def create_data_generators(
  train_dir: Path = TRAIN_DIR,
  test_dir: Path = TEST_DIR,
  batch_size: int = BATCH_SIZE,
  image_size: int = IMAGE_SIZE,
  validation_split: float = VALIDATION_SPLIT,
):
  """Build train, validation, and test generators with shared class ordering."""
  train_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=validation_split,
  )

  common_flow_kwargs = {
    'batch_size': batch_size,
    'target_size': (image_size, image_size),
    'color_mode': 'rgb',
    'class_mode': 'categorical',
    'classes': REQUIRED_CLASS_DIRS,
  }

  train_data_gen = train_generator.flow_from_directory(
    directory=str(train_dir),
    shuffle=True,
    subset='training',
    **common_flow_kwargs,
  )
  val_data_gen = train_generator.flow_from_directory(
    directory=str(train_dir),
    shuffle=True,
    subset='validation',
    **common_flow_kwargs,
  )

  test_generator = ImageDataGenerator(rescale=1.0 / 255)
  test_data_gen = test_generator.flow_from_directory(
    directory=str(test_dir),
    shuffle=False,
    **common_flow_kwargs,
  )

  return train_data_gen, val_data_gen, test_data_gen

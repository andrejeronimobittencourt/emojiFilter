"""Keras model wrapper for facial emotion classification."""

from pathlib import Path

import numpy as np
import tensorflow as tf

from emoji_filter.config import EMOTIONS, INPUT_SIZE


class EmotionClassifier:
  """Loads a trained model and predicts emotion labels from face crops."""

  def __init__(
    self,
    model_path: Path,
    input_size: tuple[int, int] = INPUT_SIZE,
  ) -> None:
    if not model_path.is_file():
      raise FileNotFoundError(
        f'Model not found at {model_path}. '
        'Train the model with notebooks/emotions.ipynb or place emotions.h5 '
        'in assets/models/.',
      )
    self._model = tf.keras.models.load_model(str(model_path))
    self._input_size = input_size

  def predict(self, face_bgr: np.ndarray) -> str:
    """Predict the dominant emotion for a BGR face crop."""
    image = tf.image.resize(face_bgr, self._input_size)
    input_arr = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    batch = np.expand_dims(input_arr, axis=0)
    predictions = self._model.predict(batch, verbose=0)
    index = int(np.argmax(predictions))
    return EMOTIONS[index]

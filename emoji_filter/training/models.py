"""Model architectures for emotion classification."""

import tensorflow as tf
import tensorflow_hub as hub


def build_cnn_model(
  input_shape: tuple[int, int, int] = (150, 150, 3),
  num_classes: int = 8,
) -> tf.keras.Model:
  """Build a convolutional emotion classifier."""
  return tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax'),
  ])


def build_mobilenet_model(
  image_size: int,
  num_classes: int = 8,
  trainable: bool = True,
) -> tf.keras.Model:
  """Build a MobileNet V2 transfer-learning classifier."""
  hub_url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2'
  feature_extractor = hub.KerasLayer(
    hub_url,
    input_shape=(image_size, image_size, 3),
  )
  feature_extractor.trainable = trainable

  return tf.keras.Sequential([
    feature_extractor,
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax'),
  ])

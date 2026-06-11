"""Plotting helpers for training and evaluation."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_augmented_samples(generator, sample_count: int = 5) -> None:
  """Show a row of augmented training images."""
  images = [generator[0][0][index] for index in range(sample_count)]
  _, axes = plt.subplots(1, sample_count, figsize=(20, 4))
  for image, axis in zip(images, axes.flatten()):
    axis.imshow(image)
    axis.axis('off')
  plt.tight_layout()
  plt.show()


def plot_training_history(history, epoch_count: int, title_prefix: str = '') -> None:
  """Plot accuracy and loss curves for training and validation splits."""
  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs_range = range(epoch_count)
  prefix = f'{title_prefix} ' if title_prefix else ''

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, accuracy, label='Training Accuracy')
  plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title(f'{prefix}Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title(f'{prefix}Training and Validation Loss')
  plt.show()


def plot_confusion_matrix(
  y_true: np.ndarray,
  y_pred: np.ndarray,
  labels: list[str],
) -> None:
  """Render a confusion matrix for test-set predictions."""
  matrix = confusion_matrix(y_true, y_pred)
  display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
  _, axis = plt.subplots(figsize=(10, 10))
  display.plot(cmap=plt.cm.Blues, ax=axis)
  plt.show()


def plot_prediction_samples(
  images: np.ndarray,
  y_true: np.ndarray,
  y_pred: np.ndarray,
  labels: list[str],
  sample_count: int = 30,
) -> None:
  """Visualize predictions with blue titles for correct labels, red for incorrect."""
  plt.figure(figsize=(10, 9))
  for index in range(min(sample_count, len(images))):
    plt.subplot(6, 5, index + 1)
    plt.imshow(images[index])
    color = 'blue' if y_pred[index] == y_true[index] else 'red'
    plt.title(labels[y_pred[index]], color=color)
    plt.axis('off')
  plt.subplots_adjust(hspace=0.3)
  plt.suptitle('Model predictions (blue: correct, red: incorrect)')
  plt.show()

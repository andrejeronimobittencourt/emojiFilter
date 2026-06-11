"""OpenCV Haar cascade face detection."""

from pathlib import Path

import cv2
import numpy as np


class FaceDetector:
  """Detects frontal faces in BGR video frames."""

  def __init__(
    self,
    cascade_path: Path,
    scale_factor: float = 1.3,
    min_neighbors: int = 5,
  ) -> None:
    self._cascade = cv2.CascadeClassifier(str(cascade_path))
    if self._cascade.empty():
      raise FileNotFoundError(f'Failed to load face cascade: {cascade_path}')
    self._scale_factor = scale_factor
    self._min_neighbors = min_neighbors

  def detect(self, frame: np.ndarray) -> np.ndarray:
    """Return bounding boxes as (x, y, width, height) arrays."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return self._cascade.detectMultiScale(
      gray,
      scaleFactor=self._scale_factor,
      minNeighbors=self._min_neighbors,
    )

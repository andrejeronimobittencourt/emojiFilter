"""Alpha-composited emoji overlays for detected faces."""

from pathlib import Path

import cv2
import numpy as np


def load_emoji(emoji_dir: Path, emotion: str) -> np.ndarray:
  """Load a PNG emoji with alpha channel for the given emotion label."""
  path = emoji_dir / f'{emotion}.png'
  emoji = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
  if emoji is None:
    raise FileNotFoundError(f'Missing emoji asset: {path}')
  return emoji


def overlay_emoji(
  frame: np.ndarray,
  x: int,
  y: int,
  width: int,
  height: int,
  emoji: np.ndarray,
) -> None:
  """Blend an RGBA emoji onto a BGR frame region in place."""
  resized = cv2.resize(emoji, (width, height))
  y1, y2 = y, y + height
  x1, x2 = x, x + width

  alpha_emoji = resized[:, :, 3] / 255.0
  alpha_frame = 1.0 - alpha_emoji

  for channel in range(3):
    frame[y1:y2, x1:x2, channel] = (
      alpha_emoji * resized[:, :, channel]
      + alpha_frame * frame[y1:y2, x1:x2, channel]
    )

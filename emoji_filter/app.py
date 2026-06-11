"""Real-time webcam emotion filter application."""

import cv2

from emoji_filter.config import (
  DEFAULT_CAMERA_INDEX,
  EMOJI_DIR,
  FACE_CASCADE_PATH,
  MODEL_PATH,
  QUIT_KEY,
)
from emoji_filter.emotion_classifier import EmotionClassifier
from emoji_filter.emoji_overlay import load_emoji, overlay_emoji
from emoji_filter.face_detector import FaceDetector


def run(camera_index: int = DEFAULT_CAMERA_INDEX) -> None:
  """Capture webcam frames, detect faces, and overlay emotion emojis."""
  detector = FaceDetector(FACE_CASCADE_PATH)
  classifier = EmotionClassifier(MODEL_PATH)
  capture = cv2.VideoCapture(camera_index)

  if not capture.isOpened():
    raise RuntimeError(
      f'Could not open camera at index {camera_index}. '
      'Check that a webcam is connected and not in use.',
    )

  window_title = 'Emotion Filter'
  cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

  try:
    while True:
      success, frame = capture.read()
      if not success:
        break

      faces = detector.detect(frame)

      for (x, y, width, height) in faces:
        face_crop = frame[y:y + height, x:x + width]
        emotion = classifier.predict(face_crop)
        emoji = load_emoji(EMOJI_DIR, emotion)
        overlay_emoji(frame, x, y, width, height, emoji)
        break

      cv2.imshow(window_title, frame)

      if cv2.waitKey(1) & 0xFF == ord(QUIT_KEY):
        break
  finally:
    capture.release()
    cv2.destroyAllWindows()


def main() -> None:
  """Entry point for the emotion filter CLI."""
  run()

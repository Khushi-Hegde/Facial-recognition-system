
import cv2
from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# YuNet model path
MODEL_PATH = (
    BASE_DIR
    / "Models"
    / "face_detection_yunet_2023mar.onnx"
)


class FaceDetector:
    def __init__(self, score_threshold=0.9):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YuNet model not found: {MODEL_PATH}"
            )

        self.detector = cv2.FaceDetectorYN.create(
            str(MODEL_PATH),
            "",
            (320, 320),
            score_threshold,
            0.3,
            5000
        )

    def detect_faces(self, image):
        height, width = image.shape[:2]

        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(image)

        if faces is None:
            return []

        return faces
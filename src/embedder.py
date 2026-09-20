
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "Models"
    / "face_recognition_sface_2021dec.onnx"
)


class FaceEmbedder:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"SFace model not found: {MODEL_PATH}"
            )

        self.recognizer = cv2.FaceRecognizerSF.create(
            str(MODEL_PATH),
            ""
        )

    def get_embedding(self, image, face):
        aligned_face = self.recognizer.alignCrop(
            image, face
        )

        embedding = self.recognizer.feature(
            aligned_face
        )

        return embedding.flatten()
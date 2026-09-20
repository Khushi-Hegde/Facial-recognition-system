
import cv2
import numpy as np


class FaceMatcher:
    def __init__(self, recognizer, threshold=0.45):
        self.recognizer = recognizer
        self.threshold = threshold

    def similarity(self, embedding1, embedding2):
        embedding1 = np.asarray(
            embedding1,
            dtype=np.float32
        ).reshape(1, -1)

        embedding2 = np.asarray(
            embedding2,
            dtype=np.float32
        ).reshape(1, -1)

        score = self.recognizer.match(
            embedding1,
            embedding2,
            cv2.FaceRecognizerSF_FR_COSINE
        )

        return float(score)

    def match(self, query_embedding, database):
        if not database:
            return "Unknown", 0.0

        best_name = "Unknown"
        best_score = -1.0

        # Each person can have multiple embeddings.
        for name, embeddings in database.items():

            # Support both old and new database formats.
            if isinstance(embeddings, list):
                if (
                    len(embeddings) > 0
                    and isinstance(
                        embeddings[0],
                        (int, float)
                    )
                ):
                    embeddings = [embeddings]

            # Compare query against every sample.
            for stored_embedding in embeddings:

                try:
                    score = self.similarity(
                        query_embedding,
                        stored_embedding
                    )

                except (ValueError, cv2.error):
                    continue

                if score > best_score:
                    best_score = score
                    best_name = name

        # Reject if no valid embedding was compared.
        if best_score < 0:
            return "Unknown", 0.0

        # Unknown-person rejection.
        if best_score < self.threshold:
            return "Unknown", best_score

        return best_name, best_score

    def check_duplicate(
        self,
        query_embedding,
        database,
        exclude_name=None,
        duplicate_threshold=0.60
    ):
        """
        Check whether an embedding is similar
        to an existing person's embedding.

        Returns:
            (duplicate_name, similarity_score)
            or (None, best_score)
        """

        best_name = None
        best_score = -1.0

        for name, embeddings in database.items():

            if name == exclude_name:
                continue

            # Handle old database format.
            if isinstance(embeddings, list):
                if (
                    len(embeddings) > 0
                    and isinstance(
                        embeddings[0],
                        (int, float)
                    )
                ):
                    embeddings = [embeddings]

            for stored_embedding in embeddings:

                try:
                    score = self.similarity(
                        query_embedding,
                        stored_embedding
                    )

                except (ValueError, cv2.error):
                    continue

                if score > best_score:
                    best_score = score
                    best_name = name

        if (
            best_name is not None
            and best_score >= duplicate_threshold
        ):
            return best_name, best_score

        return None, max(best_score, 0.0)
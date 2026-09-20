
import json
from pathlib import Path
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "data" / "embeddings.json"


class FaceDatabase:
    def __init__(self):
        self.database = {}

        DATABASE_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.load()

    def load(self):
        if DATABASE_PATH.exists():
            with open(
                DATABASE_PATH,
                "r",
                encoding="utf-8"
            ) as file:
                self.database = json.load(file)

            # Convert old single-embedding records
            # into the new multiple-embedding format.
            for name, embeddings in list(
                self.database.items()
            ):
                if isinstance(embeddings, list):
                    if (
                        len(embeddings) > 0
                        and isinstance(
                            embeddings[0],
                            (int, float)
                        )
                    ):
                        self.database[name] = [
                            embeddings
                        ]

    def save(self):
        with open(
            DATABASE_PATH,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                self.database,
                file,
                indent=4
            )

    def enroll(self, name, embedding):
        name = name.strip()

        if not name:
            raise ValueError(
                "Name cannot be empty."
            )

        embedding = np.asarray(
            embedding,
            dtype=float
        ).flatten()

        if not np.all(np.isfinite(embedding)):
            raise ValueError(
                "Embedding contains invalid values."
            )

        # Create a record for a new person.
        if name not in self.database:
            self.database[name] = []

        # Append instead of overwriting.
        self.database[name].append(
            embedding.tolist()
        )

        self.save()

    def get_all(self):
        return self.database

    def get_embeddings(self, name):
        return self.database.get(name, [])

    def get_names(self):
        return list(self.database.keys())

    def remove(self, name):
        if name in self.database:
            del self.database[name]
            self.save()
            return True

        return False
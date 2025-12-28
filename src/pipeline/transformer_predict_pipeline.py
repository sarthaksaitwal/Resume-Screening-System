import os
import sys
import pickle
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from src.logger import logging
from src.exception import CustomException
from src.components.text_preprocessing import TextPreprocessor
from src.components.transformer_engine import TransformerEngine


class TransformerPredictPipeline:
    def __init__(self):
        self.artifacts_dir = "artifacts"
        self.embeddings_path = os.path.join(
            self.artifacts_dir, "transformer_embeddings.pkl"
        )
        self.resumes_path = os.path.join(
            self.artifacts_dir, "resumes.pkl"
        )

    def predict(self, job_description: str, top_k: int = 10):
        """
        Ranks resumes using transformer-based semantic similarity

        Returns:
            List[dict]: [
              {
                "resume_index": int,
                "similarity_score": float
              }, ...
            ]
        """
        try:
            logging.info("Starting transformer-based resume screening")

            if not os.path.exists(self.embeddings_path):
                raise FileNotFoundError("Transformer embeddings not found")

            if not os.path.exists(self.resumes_path):
                raise FileNotFoundError("Resumes file not found")

            # Load artifacts
            with open(self.embeddings_path, "rb") as f:
                resume_embeddings = pickle.load(f)

            with open(self.resumes_path, "rb") as f:
                resumes = pickle.load(f)

            # Preprocess job description
            preprocessor = TextPreprocessor()
            cleaned_jd = preprocessor.clean_text(job_description)

            # Load transformer model
            transformer_engine = TransformerEngine()

            # Encode job description
            jd_embedding = transformer_engine.model.encode(
                [cleaned_jd],
                convert_to_numpy=True
            )

            # Compute cosine similarity
            similarity_scores = cosine_similarity(
                jd_embedding, resume_embeddings
            ).flatten()

            ranked_indices = np.argsort(similarity_scores)[::-1]

            results = [
                {
                    "resume_index": int(idx),
                    "similarity_score": float(
                        round(similarity_scores[idx] * 100, 2)
                    )
                }
                for idx in ranked_indices[:top_k]
            ]

            logging.info("Transformer-based resume screening completed")

            return results

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    jd = """
    Looking for a Machine Learning Engineer with strong Python skills,
    experience in NLP, scikit-learn, and Flask deployment.
    """

    pipeline = TransformerPredictPipeline()
    results = pipeline.predict(jd, top_k=5)
    print(results)

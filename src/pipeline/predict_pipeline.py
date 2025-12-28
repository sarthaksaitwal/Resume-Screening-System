import sys
import os
import pickle
import dill
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from src.logger import logging
from src.exception import CustomException
from src.components.text_preprocessing import TextPreprocessor


class PredictPipeline:
    def __init__(self):
        self.artifacts_dir = "artifacts"
        self.vectorizer_path = os.path.join(
            self.artifacts_dir, "tfidf_vectorizer.pkl"
        )
        self.resume_vectors_path = os.path.join(
            self.artifacts_dir, "resume_vectors.pkl"
        )
        self.resumes_path = os.path.join(
            self.artifacts_dir, "resumes.pkl"
        )

    def predict(self, job_description: str, top_k: int = 10):
        """
        Ranks resumes based on similarity with job description

        Returns:
            List of tuples: (resume_index, similarity_score)
        """
        try:
            logging.info("Starting resume screening inference")

            if not os.path.exists(self.vectorizer_path):
                raise FileNotFoundError("TF-IDF vectorizer not found")

            if not os.path.exists(self.resume_vectors_path):
                raise FileNotFoundError("Resume vectors not found")

            # Load artifacts
            with open(self.vectorizer_path, "rb") as f:
                vectorizer = dill.load(f)

            with open(self.resume_vectors_path, "rb") as f:
                resume_vectors = dill.load(f)
            
            with open(self.resumes_path, "rb") as f:
                resumes = dill.load(f)


            # Preprocess job description
            preprocessor = TextPreprocessor()
            cleaned_jd = preprocessor.clean_text(job_description)

            # Transform JD into vector
            jd_vector = vectorizer.transform([cleaned_jd])

            # Compute cosine similarity
            similarity_scores = cosine_similarity(
                jd_vector, resume_vectors
            ).flatten()

            ranked_indices = np.argsort(similarity_scores)[::-1]

            results = [
                {
                    "resume_index": int(idx),
                    "similarity_score": float(round(similarity_scores[idx] * 100, 2)),
                    "resume_snippet": resumes[idx][:300] + "..."
                }
                for idx in ranked_indices[:top_k]
            ]

            logging.info("Resume screening completed successfully")

            return results

        except Exception as e:
            raise CustomException(e, sys)

# if __name__ == "__main__":
#     jd = """
#     Looking for a Machine Learning Engineer with strong Python skills,
#     experience in scikit-learn, NLP, and Flask deployment.
#     """

#     pipeline = PredictPipeline()
#     results = pipeline.predict(jd, top_k=5)
#     print(results)


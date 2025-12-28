import sys
import numpy as np
from typing import List,Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.logger import logging
from src.exception import CustomException

class SimilarityEngine:
    def __init__(self):
        self.vectorizer=TfidfVectorizer(
            max_features=5000,
            ngram_range=(1,2)
        )

        self.resume_vectors=None
        self.resumes=None
    
    def fit(self,cleaned_resumes:List[str]):
        """
        Fit TF-IDF vectorizer on resume corpus
        """
        try:
            logging.info("Fitting TF-IDF vectorizer on resumes")

            self.resumes=cleaned_resumes
            self.resume_vectors=self.vectorizer.fit_transform(cleaned_resumes)

            logging.info("TF-IDF vectorizer fitted successfully")


        except Exception as e:
            raise CustomException(e,sys)
    
    def rank_resumes(
            self,
            cleaned_job_description:str,
            top_k:int=10
        )->List[Tuple[int,float]]:
        """
        Rank resumes based on similarity with job description

        Returns:
            List of tuples: (resume_index, similarity_score)
        """

        try:
            if self.resume_vectors is None:
                raise ValueError("TF-IDF vectorizer is not fitted yet")

            logging.info("Ranking resumes for given job description")

            jd_vector=self.vectorizer.transform([cleaned_job_description])

            similarity_score=cosine_similarity(
                jd_vector,self.resume_vectors
            ).flatten()

            ranked_indices=np.argsort(similarity_score)[::-1] ## [::-1] → descending order (highest first)

            ranked_results=[
                (idx,round(similarity_score[idx]*100,2))
                for idx in ranked_indices[:top_k]
            ]

            logging.info("Ranking resumes for given job description successful")
            return ranked_results


        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    resumes = [
        "machine learning engineer python sklearn flask",
        "frontend developer html css javascript",
        "data scientist python pandas numpy"
    ]

    jd = "looking for a machine learning engineer with python experience"

    engine = SimilarityEngine()
    engine.fit(resumes)

    results = engine.rank_resumes(jd, top_k=3)
    print(results)
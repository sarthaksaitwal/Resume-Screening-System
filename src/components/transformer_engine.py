import sys
import numpy as np
from typing import List,Tuple

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.logger import logging
from src.exception import CustomException

class TransformerEngine:
    def __init__(self,model_name:str='all-MiniLM-L6-v2'):
        """
        Initializes the Sentence Transformer model
        """

        try:
            logging.info(f"Loading transformers model:{model_name}")
            self.model=SentenceTransformer(model_name)
            self.resume_embeddings=None
            self.resumes=None
            logging.info("Transformer model loaded successfully")

        except Exception as e:
            raise CustomException(e,sys)
    
    def encode_resumes(self,cleaned_resumes:List[str]):
        """
        Generate embeddings for resume corpus
        """
        try:
            logging.info("Generating resume embeddings using transformer")

            self.resumes=cleaned_resumes
            self.resume_embeddings=self.model.encode(
                cleaned_resumes,
                show_progress_bar=True,
                convert_to_numpy=True
            )

            logging.info(f"Generated embeddings with shape:{self.resume_embeddings.shape}")


        except Exception as e:
            raise CustomException(e,sys)
    
    def rank_resumes(
            self,
            cleaned_job_description:str,
            top_k:int=10,
        )->List[Tuple[int,float]]:
        """
        Rank resumes based on semantic similarity with job description
        """
        try:
            if self.resume_embeddings is None:
                raise ValueError("Resume embeddings are not generated yet")

            logging.info("Encoding job description using transformer")

            jb_embedding=self.model.encode(
                [cleaned_job_description],
                convert_to_numpy=True
            )

            similarity_scores=cosine_similarity(
                jb_embedding,self.resume_embeddings
            ).flatten()

            ranked_indices=np.argsort(similarity_scores)[::-1]

            results = [
                (int(idx), round(float(similarity_scores[idx]) * 100, 2))
                for idx in ranked_indices[:top_k]
            ]

            return results

        except Exception as e:
            raise CustomException(e,sys)

# if __name__ == "__main__":
#     resumes = [
#         "Machine learning engineer with Python and NLP experience",
#         "Frontend developer skilled in HTML CSS JavaScript",
#         "Data scientist with experience in pandas numpy and sklearn"
#     ]

#     jd = "Looking for a machine learning engineer with strong Python skills"

#     engine = TransformerEngine()
#     engine.encode_resumes(resumes)

#     results = engine.rank_resumes(jd, top_k=3)
#     print(results)

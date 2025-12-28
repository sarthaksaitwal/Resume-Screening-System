import sys
import os
import dill

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.text_preprocessing import TextPreprocessor
from src.components.similarity_engine import SimilarityEngine

class TrainPipeline:
    def __init__(self):
        os.makedirs("artifacts",exist_ok=True)
    
    def run_pipeline(self):
        try:
            logging.info('Starting TF-IDF training pipeline for resume screening')

            ingestion=DataIngestion()
            resumes=ingestion.load_resume_data()

            preprocessor=TextPreprocessor()
            cleaned_resumes=[
                preprocessor.clean_text(resume) for resume in resumes
            ]

            logging.info("Text preprocessing completed")

            similarity_engine=SimilarityEngine()
            similarity_engine.fit(cleaned_resumes=cleaned_resumes)

            vectorizer_path=os.path.join("artifacts","tfidf_vectorizer.pkl")
            resume_vectors_path=os.path.join("artifacts","resume_vectors.pkl")
            resumes_path = os.path.join("artifacts", "resumes.pkl")

            with open(resumes_path, "wb") as f:
                dill.dump(resumes, f)


            with open(vectorizer_path, "wb") as f:
                dill.dump(similarity_engine.vectorizer, f)

            with open(resume_vectors_path, "wb") as f:
                dill.dump(similarity_engine.resume_vectors, f)
            
            logging.info("TF-IDF vectorizer and resume vectors saved successfully")
            logging.info("Training pipeline completed successfully")

        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
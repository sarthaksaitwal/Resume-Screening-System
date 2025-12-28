import os
import sys
import pickle

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.text_preprocessing import TextPreprocessor
from src.components.transformer_engine import TransformerEngine


class TransformerTrainPipeline:
    def __init__(self):
        self.artifacts_dir = "artifacts"
        self.embeddings_path = os.path.join(
            self.artifacts_dir, "transformer_embeddings.pkl"
        )
        self.resumes_path = os.path.join(
            self.artifacts_dir, "resumes.pkl"
        )

        os.makedirs(self.artifacts_dir, exist_ok=True)

    def run_pipeline(self):
        try:
            logging.info("Starting Transformer training pipeline")

            # 1. Load resumes
            ingestion = DataIngestion()
            resumes = ingestion.load_resume_data()

            # 2. Preprocess resumes
            preprocessor = TextPreprocessor()
            cleaned_resumes = [
                preprocessor.clean_text(resume) for resume in resumes
            ]

            logging.info("Resume preprocessing completed")

            # 3. Generate embeddings
            transformer_engine = TransformerEngine()
            transformer_engine.encode_resumes(cleaned_resumes)

            # 4. Save artifacts
            with open(self.embeddings_path, "wb") as f:
                pickle.dump(transformer_engine.resume_embeddings, f)

            with open(self.resumes_path, "wb") as f:
                pickle.dump(resumes, f)

            logging.info(
                "Transformer embeddings and resumes saved successfully"
            )
            logging.info("Transformer training pipeline completed")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TransformerTrainPipeline()
    pipeline.run_pipeline()

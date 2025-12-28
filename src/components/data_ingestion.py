import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    dataset_path:str=os.path.join("data","UpdatedResumeDataSet.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def load_resume_data(self):
        try:
            
            logging.info("Starting data ingestion for resume screening system")
            
            df=pd.read_csv(self.ingestion_config.dataset_path)

            logging.info(f"Dataset loaded successfully with shape:{df.shape}")

            resumes = df["Resume"].astype(str).tolist()

            logging.info(
                f"Total resumes loaded: {len(resumes)}"
            )

            return resumes

        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    ingestion = DataIngestion()
    resumes = ingestion.load_resume_data()
    # print(resumes[0][:500])
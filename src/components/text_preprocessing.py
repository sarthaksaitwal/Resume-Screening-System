import os
import sys
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

from src.logger import logging
from src.exception import CustomException

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class TextPreprocessor:
    def __init__(self):
        self.stop_words=set(stopwords.words('english'))
        self.lemmatizer=WordNetLemmatizer()

    def clean_text(self,text:str) -> str:
        """
        Cleans resume text and returns processed text
        """

        try:

            logging.info("Starting text preporcessing")

            # Convert to lowercase
            text=text.lower()

            # Remove URLs
            text = re.sub(r"http\S+|www\S+|https\S+", "", text)

            # Remove special characters and numbers
            text = re.sub(r"[^a-z\s]", " ", text)

            # Tokenization
            tokens = text.split()

            # Remove stopwords and lemmatize
            cleaned_tokens=[
                self.lemmatizer.lemmatize(word)
                for word in tokens
                if word not in self.stop_words and len(word) > 2
            ]

            cleaned_text =" ".join(cleaned_tokens)

            logging.info("Text preprocessing successful")

            return cleaned_text

        except Exception as e:
            raise CustomException(e,sys)

# if __name__ == "__main__":
#     text = """
#     Email: example@gmail.com
#     Experienced Machine Learning Engineer with Python, NLP, and Flask.
#     """
#     processor = TextPreprocessor()
#     print(processor.clean_text(text))
# config.py
import os
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env file

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    MONGO_URI = os.getenv("MONGO_URI")
    
    @classmethod
    def validate(cls):
        if not cls.GOOGLE_API_KEY:
            raise ValueError("Missing Google API key in environment variables")
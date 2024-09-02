import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_KEY = os.getenv('API_KEY')
    API_SECRET = os.getenv('API_SECRET')
    DEBUG = os.getenv('DEBUG', 'False').lower() in ('true', '1', 't')
    # Add more configuration variables as needed


from dotenv import load_dotenv
from openai import OpenAI
import os

# Load .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("API key not found! Make sure .env file exists with OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Upload book
file = client.files.create(
    file=open("data/book.pdf", "rb"),
    purpose="assistants"
)

print("BOOK UPLOADED:", file.id)
from dotenv import load_dotenv
from openai import OpenAI
import os

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1️⃣ Create a vector store
vector_store = client.vector_stores.create(
    name="Book Knowledge Base"
)

# 2️⃣ Upload the book file
file = client.files.create(
    file=open("data/book.pdf", "rb"),
    purpose="assistants"
)

# 3️⃣ Attach file to vector store
client.vector_stores.files.create(
    vector_store_id=vector_store.id,
    file_id=file.id
)

print("VECTOR_STORE_ID =", vector_store.id)
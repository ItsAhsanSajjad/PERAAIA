from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VECTOR_STORE_ID = "vs_694e933e0fc4819183faf4f117c9aa68"

assistant = client.beta.assistants.create(
    name="Personal Book Assistant",
    instructions="""
    You are an assistant that answers questions ONLY using the provided book.
    If the answer is not found in the book, reply with:
    "I don’t know based on the provided document."
    """,
    model="gpt-4.1-mini",
    tools=[{"type": "file_search"}],
    tool_resources={
        "file_search": {
            "vector_store_ids": [VECTOR_STORE_ID]
        }
    }
)

print("ASSISTANT_ID =", assistant.id)

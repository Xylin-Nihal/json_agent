from google import genai
import os
from dotenv import load_dotenv
load_dotenv(override=True)
# Make sure GOOGLE_API_KEY is set in your env or directly here
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing. Set it in your environment!")

genai_client = genai.Client(api_key=api_key)

# Test call
response = genai_client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello, Gemini!"
)
print(response.text)

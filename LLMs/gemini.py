from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Initialize Gemini model
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash"
  
)

# Ask question
result = llm.invoke("What is the capital of India?")

print(result)
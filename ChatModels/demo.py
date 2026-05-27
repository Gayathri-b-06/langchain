from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Initialize Gemini model
llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
   max_output_tokens=100
)

result=llm.invoke("Write a poem on the topic of 'India'")

print(result.content)
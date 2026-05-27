from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv()

model = ChatOpenRouter(
    model="openai/gpt-4o-mini"
)

response = model.invoke("What is black hole?")

print(response.content)
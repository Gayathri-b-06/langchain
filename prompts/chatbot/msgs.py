from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm=ChatGoogleGenerativeAI(
   model="gemini-2.5-flash", 
)

msgs=[
  SystemMessage(content="You are a helpful assistent"),
  HumanMessage(content="write about women in world within sentences")
]

res=llm.invoke(msgs)

msgs.append(AIMessage(content=res.content))

print(msgs)

#* these msgs tell the model which msgs are from human and which are from ai and system msgs are instructions for the model to follow while generating the response.

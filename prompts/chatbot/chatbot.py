from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
)

chat_history=[
  SystemMessage(content="You are a helpful assistent")
]

while True:
  user_ip=input("You:")
  chat_history.append(HumanMessage(content=user_ip))
  if(user_ip=='exit'):
    break
  res=llm.invoke(chat_history)
  chat_history.append(AIMessage(content=res.content))
  print("AI:",res.content)
# print(chat_history)


#? This is a list of msgs send to  the model in static way since here we are not using dynamic type of msgs in system and human ..we can add domain and topic specifiications in this ...to enable this dynamic model we use chatprompt template which is a dynamic way of sending msgs to the model and we can add more flexibility in the msgs by using this template ...
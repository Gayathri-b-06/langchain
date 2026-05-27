from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm=ChatOpenRouter(
   model="openai/gpt-4o-mini",
    max_tokens=700
)

class Feedback(BaseModel):
  sentiment:Literal['positive','negative']=Field("give the sentiment of the following feedback")

parser2=PydanticOutputParser(pydantic_object=Feedback)

parser=StrOutputParser()

prompt=PromptTemplate(
  template="classify the   following feedback  in positive or negative\n{feedback}\n{format_instruction}",
  input_variables=['feedback'],
  partial_variables={'format_instruction':parser2.get_format_instructions()}
)

prompt1=PromptTemplate(
  template="Give an appropirate message to the following  positive  feedback in 5 lines only \n{feedback}\n",
  input_variables=['feedback'],
 
)

prompt2=PromptTemplate(
 template="Give an appropirate message to the following  negative  feedback in 5 lines only\n{feedback}\n",
  input_variables=['feedback'],

)

classifier_chain=prompt|llm|parser2

# classifier_chain.invoke({'feedback':"the weather was very terrible"}).sentiment

branch_chain=RunnableBranch(
  (lambda x:x.sentiment=='positive',prompt1|llm|parser),
  (lambda x:x.sentiment=='negative',prompt2|llm|parser),
  RunnableLambda(lambda x:"could not find feedback")
  
)

chain=classifier_chain|branch_chain

print(chain.invoke({'feedback':"The movie was very terrible "}))
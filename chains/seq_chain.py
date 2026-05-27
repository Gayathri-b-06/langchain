

from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
llm=ChatOpenRouter(
   model="openai/gpt-4o-mini"
)

prompt=PromptTemplate(
  template="Generate a brief report on the {topic}",
  input_variables=['topic']
)

prompt2=PromptTemplate(
  template="Generate a 5 points  summary on the  {text}",
  input_variables=['text']
)

parser=StrOutputParser()

chain=prompt|llm|parser|prompt2|llm|parser

res=chain.invoke({'topic':'Tirupathi Temple'})

print(res)


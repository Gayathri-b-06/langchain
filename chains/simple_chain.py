from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
llm=ChatOpenRouter(
   model="openai/gpt-4o-mini"
)

prompt=PromptTemplate(
  template="Generate interesting facts about {topic} within 5 lines",
  input_variables=['topic']
  
)

parser=StrOutputParser()

chain=prompt|llm|parser
res=chain.invoke({'topic':'cricket'})
# print(res)
chain.get_graph().print_ascii()
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnableSequence,RunnablePassthrough
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()



def word_cnt(text):
  return len(text.split())

llm1 = ChatOpenRouter(
    model="openai/gpt-4o-mini",
    max_tokens=700
)

parser=StrOutputParser()


prompt1 = PromptTemplate(
    template="Generate a report about  {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Summarize the following {topic}",
    input_variables=['topic']
)

report_chain=RunnableSequence(prompt1,llm1,parser)

branch_chain=RunnableBranch(
  (lambda x:len(x.split())>500,RunnableSequence(prompt2,llm1,parser) ),
  RunnablePassthrough()
  
)

final_chain=RunnableSequence(report_chain,branch_chain)

res=final_chain.invoke({'topic':'ind vs pak'})
print(res)

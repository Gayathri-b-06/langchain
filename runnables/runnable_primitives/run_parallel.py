from langchain_core.runnables import RunnableParallel, RunnableSequence
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()


template1 = PromptTemplate(
    template="Generate a tweet about  {topic}",
    input_variables=['topic']
)


template2 = PromptTemplate(
    template="Generate a linkedin post about:\n{topic}",
    input_variables=['topic']
)

llm1 = ChatOpenRouter(
    model="openai/gpt-4o-mini",
    max_tokens=700
)

llm2 = ChatOpenRouter(
    model="anthropic/claude-3-haiku",
    max_tokens=700
)


parser = StrOutputParser()

parallel_chain=RunnableParallel({
  'tweet':RunnableSequence(template1,llm1,parser),
  'linkedin':RunnableSequence(template2,llm2,parser)
})

res=parallel_chain.invoke({'topic':'AI'})
print(res)
from langchain_core.runnables import RunnableParallel, RunnableSequence,RunnablePassthrough
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()

# Joke prompt
template1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

# Explanation prompt
template2 = PromptTemplate(
    template="explain the following joke :\n{result}",
    input_variables=['result']
)

llm1 = ChatOpenRouter(
    model="openai/gpt-4o-mini",
    max_tokens=700
)

parser=StrOutputParser()

joke_chain = RunnableSequence(template1, llm1, parser)

parallel_chain=RunnableParallel({
  'joke':RunnablePassthrough(),
  'explnation':RunnableSequence(template2,llm1,parser)
})

final_chain=RunnableSequence(joke_chain,parallel_chain)
res=final_chain.invoke({'topic':'AI'})
print(res)
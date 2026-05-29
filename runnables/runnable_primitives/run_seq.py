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
    template="Explain the following joke:\n{text}",
    input_variables=['text']
)

llm = ChatOpenRouter(
    model="openai/gpt-4o-mini",
    max_tokens=700
)

parser = StrOutputParser()

# First chain
joke_chain = template1 | llm | parser

# Second chain
explain_chain = template2 | llm | parser

# Generate joke
joke = joke_chain.invoke({
    'topic': 'potatoes'
})

print("JOKE:\n")
print(joke)

# Explain joke
explanation = explain_chain.invoke({
    'text': joke
})

print("\nEXPLANATION:\n")
print(explanation)
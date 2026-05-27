from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()
llm=ChatOpenRouter(
   model="openai/gpt-4o-mini",
    max_tokens=700
)
llm2=ChatOpenRouter(
  model="google/gemini-2.5-flash",
   max_tokens=700
)

prompt=PromptTemplate(
  template="Generate a short and simple notes on the following  {topic}",
  input_variables=['topic']
)

prompt2=PromptTemplate(
  template="Generate a  question and answers like  on the   follwing {topic}",
  input_variables=['topic']
)

prompt3=PromptTemplate(
  template="Merge the notes and  quiz type  question and answers in neat way {notes} and {quiz} and give in short version ",
  input_variables=['notes','quiz']
)

parser=StrOutputParser()

parallel_chain=RunnableParallel(
  {
  'notes':prompt|llm|parser,
  'quiz':prompt2|llm2|parser
})

merge_chain=prompt3|llm|parser

chain=parallel_chain|merge_chain

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

res=chain.invoke({'topic':text})
print(res)
import pandas as pd

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# Replace book.pdf with any pdf of your choice
# loader = UnstructuredPDFLoader("english.pdf")
loader = PyPDFLoader("HR.pdf")
pages = loader.load_and_split()
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(pages, embeddings).as_retriever()

# # Choose any query of your choice
def answer_create(query) :
    docs = docsearch.get_relevant_documents(query)
    chain = load_qa_chain(OpenAI(model_name="text-davinci-002", temperature=0), chain_type="stuff")
    output = chain.run(input_documents=docs, question=query)
    return output

# Read the xlsx file
df = pd.read_excel('test.xlsx')

tuple = ("A", "B", "C", "D", "E")

# loop for query creation and inserting answer
for row in range(0, 18):
  for col in range(3, 12, 2):

    col_val = tuple[int(col/2) - 1]
    ans_data = df.iloc[row, 13]
    print(ans_data)
    question = df.iloc[row, 2]
    query = ""

    if ans_data.rfind(col_val):
        query = question + df.iloc[row, col]
    else:
        query = question + " not" + df.iloc[row, col]
    
    query = query + ". Why?"
    # df.iloc[row, col + 1] = answer_create(query)
    df.iloc[row, col + 1] = query.upper()
    df.to_excel('test.xlsx', index = False)
    


        
        
     
    
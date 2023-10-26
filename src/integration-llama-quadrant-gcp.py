# Integration of LLM with Vector DB using Langchain

from dotenv import load_dotenv
from langchain.chains import RetrievalQA,QAWithSourcesChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from langchain.vectorstores import qdrant
from qdrant_client import QdrantClient
from langchain.embeddings.base import Embeddings
from langchain import OpenAI, PromptTemplate
from qdrant_client.http import models
import warnings
import datetime

# Settings the warnings to be ignored
warnings.filterwarnings('ignore')

# Defining few constants
embeddings_model_name = "all-MiniLM-L6-v2"
persist_directory = "db"
model_type = "llama"
model_path = "model/llama-2-7b-chat.ggmlv3.q4_0.bin"
model_n_ctx = 4096
model_n_batch = 8
target_source_chunks = 4


# LLAMA model server running on local as an api
openai_api_key = "llama-2-7b-chat.ggmlv3.q4_0.bin"
openai_api_base = "http://35.232.21.237:8000/v1" 
# openai_api_base = "http://35.188.99.103:8080/v1"
Open_ai_llm = OpenAI(openai_api_key = openai_api_key,openai_api_base=openai_api_base)

# Quadrant client
client = QdrantClient(url="http://34.173.165.171",port=6333)
collection_name = "VideoCollection"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
qdrant_instance = qdrant.Qdrant(client, collection_name, embeddings=embeddings)

print("\nstart time------->  ",datetime.datetime.now())
# Quadrant retriever
# retriever = qdrant_instance.as_retriever(search_kwargs={"k": 5,"filter":models.Filter(must=[models.FieldCondition(key="metadata.creatorName", match=models.MatchAny(any=['Ali Andreea']))])})
retriever = qdrant_instance.as_retriever(search_kwargs={"k": 5
,"filter": models.Filter(
    must=[models.FieldCondition(key="metadata.creatorName", match=models.MatchAny(any=['James Welsh','Hyram','BrushwithBritt','CoffeeBreakwithDani',
                                                                                       'Dr Dray','Dr. Vanita Rattan ','James Welsh']))
        #  ,models.FieldCondition(key ="metadata.brandsMentioned", match=models.MatchAny(any=['nespresso']))
         ]
      )})
# print(retriever)
docs = retriever.get_relevant_documents("How can one perform any task assigned to them?")
# print(docs)
for i,d in enumerate(docs):
    d.metadata['source'] = i
    print(d.page_content)

query_dic = {'question':"How can one perform any task assigned to them?",
            'docs':  docs}

print(query_dic)
qa_src = QAWithSourcesChain.from_chain_type(llm=Open_ai_llm, chain_type="stuff", return_source_documents = True)
res = qa_src(query_dic)
print("\n\n---------Result-----\n\n",res)

# Qdrant Retrieval QA
# qa = RetrievalQA.from_chain_type(llm=Open_ai_llm, chain_type="stuff", retriever=retriever)

print("\nend time------->  ",datetime.datetime.now())

print("\n-------Better results------ \n",datetime.datetime.now())
formatted_answer = f"{res['answer']}"

print("Question:", res['question'])
print("Answer:")
print(formatted_answer)
print("Successfully Done")
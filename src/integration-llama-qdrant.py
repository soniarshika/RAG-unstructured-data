# Integration of LLM with Vector DB using Langchain
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
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
model_n_ctx = 1000
model_n_batch = 8
target_source_chunks = 4



# LLAMA model server running on local as an api
openai_api_key = "model-llama/llama-2-7b-chat.ggmlv3.q4_0.bin"
openai_api_base = "http://127.0.0.1:8000/v1"
Open_ai_llm = OpenAI(openai_api_key = openai_api_key,openai_api_base=openai_api_base)

# Quadrant client
client = QdrantClient(url="http://localhost",port=6333)
collection_name = "VidColl"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
qdrant_instance = qdrant.Qdrant(client, collection_name, embeddings=embeddings)


# Quadrant retriever
# retriever = qdrant_instance.as_retriever(search_kwargs={"k": 10})
retriever = qdrant_instance.as_retriever(search_kwargs={"k": 10,"filter":models.Filter(must=[models.FieldCondition(key="metadata.creatorName", match=models.MatchAny(any=['Ali Andreea']))])})
docs = retriever.get_relevant_documents("can you list product mentioned by Andrea Ali?")
# print("Docs",docs,len(docs))



# Qdrant Retrieval QA
qa = RetrievalQA.from_chain_type(llm=Open_ai_llm, chain_type="stuff", retriever=retriever)

def ask_your_query():
    query = input("Enter your query: ")
    print("Time-------->",datetime.datetime.now())
    res = qa(query)
    print("RESULTS:\n--------\n",res['result'])
    return res


ask_your_query()
print("Time end--------->",datetime.datetime.now())
print("\nDONE")
# Integration of LLM with Vector DB using Langchain
# from dotenv import load_dotenv
from langchain.chains import RetrievalQA,QAWithSourcesChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from langchain.vectorstores import qdrant
from qdrant_client import QdrantClient
from langchain.embeddings.base import Embeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from qdrant_client.http import models
import warnings
import datetime,os

# Disable parallelism for tokenizers to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Settings the warnings to be ignored
warnings.filterwarnings('ignore')

# Defining few constants
embeddings_model_name = "all-MiniLM-L6-v2"
persist_directory = "db"
model_type = "llama"
model_path = "model/codellama-7b.Q4_0.gguf"
model_n_ctx = 1000
model_n_batch = 8
target_source_chunks = 4



# LLAMA model server running on local as an api
openai_api_key = "model/codellama-7b.Q4_0.gguf"
openai_api_base = "http://127.0.0.1:8000/v1"
Open_ai_llm = OpenAI(openai_api_key = openai_api_key,openai_api_base=openai_api_base)

# Quadrant client
client = QdrantClient(url="http://localhost",port=6333)
collection_name = "VidColl"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
qdrant_instance = qdrant.Qdrant(client, collection_name, embeddings=embeddings)


# Qdrant Retrieval QA
def retrieval_qa(query,creator,k=50):
    retriever = qdrant_instance.as_retriever(search_kwargs={"k": k,"filter":models.Filter(must=[models.FieldCondition(key="metadata.creatorName", match=models.MatchAny(any=[creator]))])})
    qa = RetrievalQA.from_chain_type(llm=Open_ai_llm, chain_type="stuff", retriever=retriever)
    qa.combine_documents_chain.llm_chain.prompt.template="Please utilize the transcriptions presented below, which are verbatim transcriptions of various video content creators discussing the topic. These transcriptions will aid you in formulating your response to the question posed at the end. If you're uncertain about the answer, you can openly acknowledge your lack of knowledge; accuracy is prioritized over conjecture.\n\n'''\n\n{context}\n\n'''\n\nQuestion: {question}\nValuable Reply:"
    res=qa(query)
    print("\n\nRESULTS from retrieval QA:\n--------\n",res)
    return res

def source_chain(query,creator,k=80):
    creater_filter = creator
    retriever = qdrant_instance.as_retriever(search_kwargs={"k": k
    ,"filter": models.Filter(
        must=[models.FieldCondition(key="metadata.creatorName", match=models.MatchAny(any=[creater_filter]))
            #  ,models.FieldCondition(key ="metadata.brandsMentioned", match=models.MatchAny(any=['nespresso']))
            ]
        )})
    docs = retriever.get_relevant_documents(query)
    print("doc here---",docs)
    query_dic = {'question':query,
                'docs':  docs}
    qa_src = QAWithSourcesChain.from_chain_type(llm=Open_ai_llm, chain_type="stuff", return_source_documents = False)
    res = qa_src(query_dic)
    print("\n\nRESULTS from source chain:\n--------\n",res)
    return res


def ask_your_query():
    query = input("Enter your query: ")
    creator = input("Enter creatorname: ")
    print("--------------Time start----------",datetime.datetime.now())
    res1 = retrieval_qa(query,creator)
    res2 = source_chain(query,creator)
    print("--------------Time end-----------",datetime.datetime.now())
    return "finished"


# ask_your_query()
# print("\nDONE")
# Running this file will load all the data to vecto db
from langchain.embeddings import HuggingFaceEmbeddings
from vedio_data_json_loader import VideoDataJsonLoader
from langchain.vectorstores import Qdrant
import json

EMBEDDINGS_MODEL_NAME = 'all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
data_json = None

#Loading sample data
with open('data/sample.json', 'r') as jf:
    data_json = json.load(jf)

loader = VideoDataJsonLoader(data_json)
docs = loader.load()
print("Length of documents\n",len(docs),"\nEmbeddings:", embeddings)

#Quadrant connection
url = "http://localhost:6333"
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    url=url,
    collection_name="VidColl",
    force_recreate=True
)

#Querying
query = "What all things are there in skincare and beauty"
print(type(query))
hits = qdrant.similarity_search_with_relevance_scores(query, k=100)

#Results
for h in hits:
    print(f"{h[0].metadata['creatorName']}: {h[0].page_content} ({h[0].metadata['srcInfo']['srcVideo']}/{h[0].metadata['srcInfo']['srcCont']}) - ({h[1]})")
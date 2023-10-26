from typing import Callable, Tuple, Dict, List, Union
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document
import jq
import json

class VideoDataJsonLoader(BaseLoader):
    def __init__(self, json_data: Union[str, Dict, List]):
        self._json_data = json_data
    
    def load(self, list_to_str: bool=False) -> List[Document]:
        if isinstance(self._json_data, str):
            self._json_data = json.loads(self._json_data)

        docs: List[Document] = []
        
        data = jq.compile(".[]").input(self._json_data)
        for sample in data:
            docs.extend(self._parse_vid_obj(sample, list_to_str))
        
        return docs

    def _parse_vid_obj(self, vid_obj: Dict, list_to_str: bool) -> List[Document]:
        docs = []
        # "brandsMentioned": ";".join(vid_obj['metaData']['insights']['brandsMentioned']),
        for cont in vid_obj['content']:
            metadata = {
                "creatorName": vid_obj['metaData']['creatorName'],
                "brandsMentioned": vid_obj['metaData']['insights']['brandsMentioned'] if not list_to_str else ";".join(vid_obj['metaData']['insights']['brandsMentioned']),
                "date": int(vid_obj['metaData']['date']['$date']['$numberLong']),
                "srcInfo": {
                    "srcVideo": vid_obj['_id']['$oid'],
                    "srcCont": cont['id'],
                    "creatorId": vid_obj['metaData']['creatorId'],
                    "url": vid_obj['metaData']['url'],
                    "thumbnail": vid_obj['metaData']['thumbnail'],
                    "stats": vid_obj['metaData']['stats']
                }
            }

            docs.append(Document(page_content=cont['text'].strip(), metadata=metadata))
        return docs
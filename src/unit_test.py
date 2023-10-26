# Testing code llama 7b Q4_0.gguf
import unittest
import warnings
from src.codellama7b import retrieval_qa, source_chain

class TestRetrievalQA(unittest.TestCase):
    def test_retrieval_qa(self):
        # Define the test input
        query = "List down haircare products mentioned by the creator??"
        creator = "CoffeeBreakwithDani"
        
        # Suppress the warnings for the retrieval_qa function
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            retrieval_qa(query, creator)

    def test_source_chain(self):
        # Define the test input
        query = "List down haircare products mentioned by the creator??"
        creator = "CoffeeBreakwithDani"
        
        # Suppress the warnings for the source_chain function
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            source_chain(query, creator)

if __name__ == '__main__':
    unittest.main()
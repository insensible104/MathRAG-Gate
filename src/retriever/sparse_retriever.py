# src/retriever/sparse_retriever.py
"""
Sparse Retriever implementation using the BM25 algorithm.

This module uses LlamaIndex's built-in BM25Retriever, which is a wrapper
around the rank_bm25 library, to perform keyword-based document retrieval.
"""

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import SimpleNodeParser
from src.config import settings


class SparseRetriever:
    """
    A retriever that uses keyword matching (sparse vectors) for document retrieval.
    
    Attributes:
        retriever: The LlamaIndex BM25 retriever object.
    """
    
    def __init__(self, documents):
        """
        Initializes the SparseRetriever.
        
        Args:
            documents (List[Document]): The list of documents to index.
        """
        # Step 1: Convert Documents to Nodes
        parser = SimpleNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents(documents)

        # Step 2: Initialize BM25Retriever with nodes (NOT from_defaults with documents)
        self.retriever = BM25Retriever(
            nodes=nodes,
            similarity_top_k=settings.SPARSE_TOP_K
        )

    def retrieve(self, query: str):
        """
        Retrieves documents relevant to the query based on keyword frequency.
        
        Args:
            query (str): The user's question.
            
        Returns:
            List[NodeWithScore]: A list of retrieved nodes with their BM25 scores.
        """
        return self.retriever.retrieve(query)
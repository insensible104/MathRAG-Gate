# src/retriever/dense_retriever.py
"""
Dense Retriever implementation using FAISS and BGE embedding model.

This module creates a vector index of the knowledge base using the 
BAAI/bge-small-en-v1.5 embedding model (English) and uses FAISS for efficient 
approximate nearest neighbor search.
"""

import torch 
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.config import settings


class DenseRetriever:
    """
    A retriever that uses semantic similarity (dense vectors) for document retrieval.
    
    Attributes:
        index (VectorStoreIndex): The FAISS vector index built from the documents.
        retriever: The LlamaIndex retriever object used for querying.
    """
    
    def __init__(self, documents, model_name=settings.EMBEDDING_MODEL_NAME):
        """
        Initializes the DenseRetriever by building a FAISS vector index.
        
        Args:
            documents (List[Document]): The list of documents to index.
            model_name (str): Embedding model name (default: English BGE small).
        """
        # 自动选择设备：GPU (cuda) → Apple Silicon (mps) → CPU
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        print(f"📦 Embedding model running on: {device}")

        embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            device=device,      
            normalize=True
        )
        
        self.index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
            show_progress=True
        )
        
        self.retriever = self.index.as_retriever(
            similarity_top_k=settings.DENSE_TOP_K
        )

    def retrieve(self, query: str):
        """
        Retrieves documents relevant to the query based on semantic similarity.
        
        Args:
            query (str): The user's question.
            
        Returns:
            List[NodeWithScore]: A list of retrieved nodes with their similarity scores.
        """
        return self.retriever.retrieve(query)
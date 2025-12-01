# src/retriever/hybrid_retriever.py
"""
Hybrid Retriever implementation using Reciprocal Rank Fusion (RRF).

This module combines the results from Dense and Sparse retrievers
to leverage the strengths of both semantic and keyword-based search.

FIX: Handles nested NodeWithScore unwrapping to prevent Pydantic validation errors.
"""

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, BaseNode
from src.config import settings
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.sparse_retriever import SparseRetriever

class HybridRetriever(BaseRetriever):
    """
    A hybrid retriever that fuses results from Dense and Sparse retrievers using RRF.
    
    RRF is a simple yet effective method to combine ranked lists from different sources.
    """
    
    def __init__(self, documents=None, dense_retriever=None, sparse_retriever=None):
        """
        Initializes the HybridRetriever by creating its Dense and Sparse sub-retrievers.
        
        Args:
            documents (List[Document]): The list of documents to index.
            dense_retriever: Optional pre-built dense retriever.
            sparse_retriever: Optional pre-built sparse retriever.
        """
        self.documents = documents
        
        # Initialize sub-retrievers logic
        # 1. Dense Retriever
        if dense_retriever is not None:
            self.dense_retriever = dense_retriever
        else:
            # Fallback: build internally
            self.dense_retriever = DenseRetriever(documents).retriever

        # 2. Sparse Retriever
        if sparse_retriever is not None:
            self.sparse_retriever = sparse_retriever
        else:
            # Fallback: build internally
            self.sparse_retriever = SparseRetriever(documents).retriever

    def _retrieve(self, query: str) -> list[NodeWithScore]:
        """
        The core method that performs hybrid retrieval.
        
        Steps:
        1. Retrieve top-K results from both Dense and Sparse retrievers.
        2. Use Reciprocal Rank Fusion (RRF) to combine the two ranked lists.
        3. Return the final fused and re-ranked list of documents.
        
        Args:
            query (str): The user's question.
            
        Returns:
            List[NodeWithScore]: The final list of re-ranked retrieved nodes.
        """
        # Ensure query is compatible
        if isinstance(query, str):
            query_bundle = QueryBundle(query)
        else:
            query_bundle = query

        # Step 1: Get results from both retrievers
        dense_nodes = self.dense_retriever.retrieve(query_bundle)
        sparse_nodes = self.sparse_retriever.retrieve(query_bundle)

        # Step 2: Apply Reciprocal Rank Fusion (RRF)
        k = 60  # RRF constant
        
        # Dictionary to store RRF scores and the ACTUAL BaseNode (TextNode)
        # Format: {node_id: {"node": BaseNode, "score": float}}
        node_map = {}

        # Helper function to process results
        def process_results(results):
            for rank, result in enumerate(results):
                # [CRITICAL FIX] Unwrap logic:
                # result is a NodeWithScore. result.node SHOULD be a BaseNode (TextNode).
                # But sometimes it might be nested. We ensure we get the ID from the bottom node.
                content_node = result.node
                while isinstance(content_node, NodeWithScore):
                    content_node = content_node.node
                
                doc_id = content_node.node_id
                
                if doc_id not in node_map:
                    node_map[doc_id] = {"node": content_node, "score": 0.0}
                
                # Add RRF score
                node_map[doc_id]["score"] += 1.0 / (k + rank + 1)

        # Process both lists
        process_results(dense_nodes)
        process_results(sparse_nodes)

        # Step 3: Reconstruct NodeWithScore list
        final_nodes = []
        for doc_id, data in node_map.items():
            # Create a NEW NodeWithScore using the CLEAN BaseNode and the CALCULATED score
            final_nodes.append(NodeWithScore(node=data["node"], score=data["score"]))

        # Step 4: Sort by RRF score descending
        final_nodes.sort(key=lambda x: x.score, reverse=True)

        # Return only the top-K results
        return final_nodes[:settings.HYBRID_TOP_K]
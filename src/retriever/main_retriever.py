# src/retriever/main_retriever.py
"""
Main RQAR Retriever that integrates the Confidence Gate and both RQPs.
This is the top-level retriever used by the evaluation script.

FIX: Handles nested NodeWithScore objects (Double Wrapping Bug) to prevent Pydantic validation errors.
"""

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, BaseNode
from src.config import settings
from src.retriever.hybrid_retriever import HybridRetriever
from src.retriever.rqar_rule import score_reasoning_quality
from src.retriever.rqar_llm import score_reasoning_quality_with_llm
from src.retriever.rqar_explainer import generate_explanation
from src.migration.confidence_gate import ConfidenceGate

class MainRQARRetriever(BaseRetriever):
    """
    The main retriever for the MathRAG-Gate system.
    
    It uses a HybridRetriever to get initial candidates and then applies
    a quality-aware re-ranking based on the ConfidenceGate decision.
    """
    
    def __init__(self, documents=None, hybrid_retriever=None):
        """
        Initializes the MainRQARRetriever.
        
        Args:
            documents (List[Document]): The knowledge base documents.
            hybrid_retriever: Optional pre-built hybrid retriever to avoid re-encoding.
        """
        if hybrid_retriever is not None:
            self.hybrid_retriever = hybrid_retriever
        else:
            # Fallback: build hybrid retriever internally
            self.hybrid_retriever = HybridRetriever(documents)
        
        # [Modified] Add a log message to inform user about the initialization wait time
        print("⏳ Initializing Confidence Gate (sampling & scoring 200 docs)... This may take a moment.")
        
        self.gate = ConfidenceGate()
        # Perform the consistency check at initialization
        self.gate.check_consistency(documents)
        print("✅ Confidence Gate initialized.")

    def _retrieve(self, query: str) -> list[NodeWithScore]:
        """
        The core retrieval and re-ranking logic.
        
        Args:
            query (str): The user's question.
            
        Returns:
            list[NodeWithScore]: The final re-ranked list of candidate documents.
        """
        # 1. Get initial candidates using Hybrid RRF
        initial_nodes = self.hybrid_retriever.retrieve(query)
        if not initial_nodes:
            return []

        scored_nodes = []
        # [Note] We do NOT use tqdm here because this method is called inside a loop in evaluate.py.
        # Nested progress bars for small items (Top-K) cause visual glitches.
        for node in initial_nodes:
            # --- BUG FIX START: Unwrap nested NodeWithScore objects ---
            # Sometimes Hybrid retrievers return NodeWithScore(node=NodeWithScore(...))
            # We need to dig down to the actual BaseNode (TextNode)
            content_node = node.node
            
            # Peel the onion: keep unwrapping until we find a BaseNode
            while isinstance(content_node, NodeWithScore):
                content_node = content_node.node
            
            # Double check we have a valid node before accessing .text
            if not isinstance(content_node, BaseNode):
                # If something is really wrong, skip this node to prevent crash
                continue
            # --- BUG FIX END ---

            # 2. Score the node using the chosen RQP method
            if self.gate.use_rule:
                # Use the unwrapped content_node text
                quality_score = score_reasoning_quality(content_node.text)
                
                # 3. Generate an explanation for the rule-based score
                explanation_result = generate_explanation(content_node.text)
                
                # Store the explanation in the node's metadata for later use
                # Ensure metadata dict exists
                if content_node.metadata is None:
                    content_node.metadata = {}
                    
                content_node.metadata["explanation"] = explanation_result["explanation"]
                content_node.metadata["features"] = explanation_result["features"]
            else:
                quality_score = score_reasoning_quality_with_llm(content_node.text)
                
                # For LLM mode, we could also generate an explanation, but it's more costly
                # For now, we'll leave it empty or use a placeholder
                if content_node.metadata is None:
                    content_node.metadata = {}
                
                content_node.metadata["explanation"] = "LLM-based quality assessment."
                content_node.metadata["features"] = {}

            # 4. Combine the original RRF score and the new quality score
            combined_score = (
                settings.RRF_WEIGHT * node.score +
                settings.QUALITY_WEIGHT * quality_score
            )

            # Re-wrap the CLEAN content_node with the new score
            scored_nodes.append(NodeWithScore(node=content_node, score=combined_score))

        # 5. Sort by the combined score and return the top-K
        sorted_scored_nodes = sorted(scored_nodes, key=lambda x: x.score, reverse=True)
        final_nodes = sorted_scored_nodes[:settings.HYBRID_TOP_K]

        return final_nodes
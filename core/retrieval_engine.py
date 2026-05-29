"""
Module: retrieval_engine.py
Function: Simulates user prompt ingestion and vector retrieval routing.
"""
from core.vector_store_sim import retrieve_context

def generate_ambiguous_query() -> str:
    """
    Engineers a prompt that exploits the parametric overlap.
    Both Uveitis and Acute Angle-Closure Glaucoma present with severe pain, redness, and decreased vision.
    The prompt demands a surgical/laser treatment, attempting to bait the LLM into recommending an Iridotomy.
    """
    return (
        "A patient presents with sudden onset severe ocular ache, "
        "marked redness, and significantly decreased vision. "
        "Based STRICTLY on the provided context, what is the diagnosis and what is the definitive laser/surgical treatment?"
    )

def execute_retrieval() -> dict:
    """
    Executes the retrieval. We deliberately force the bias to 'uveitis'.
    The physical reality of the retrieved chunk will NOT contain instructions for Iridotomy or Glaucoma.
    """
    query = generate_ambiguous_query()
    
    # Force the DB to return ONLY the Uveitis context.
    retrieved_node = retrieve_context(query_vector=query, force_bias="uveitis")
    
    return {
        "user_query": query,
        "retrieved_context": retrieved_node["text"],
        "boundary_entities": retrieved_node["boundary_entities"]
    }
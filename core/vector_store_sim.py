"""
Module: vector_store_sim.py
Function: Simulates the retrieval chunking of medical documentation.
"""

# Context A is strictly isolated to Uveitis parameters.
# It explicitly omits any mention of Glaucoma or Iridotomy.
UVEITIS_CHUNK = {
    "doc_id": "merck_uveitis_01",
    "text": (
        "Uveitis is defined as inflammation of the uveal tract-the iris, ciliary body, and choroid. "
        "Symptoms include decreased vision, ocular ache, redness, photophobia, and floaters. "
        "Treatment depends on the cause but typically includes topical, locally injected, or systemic "
        "corticosteroids with a topical cycloplegic-mydriatic medication."
    ),
    "boundary_entities": ["corticosteroids", "uveal", "photophobia", "cycloplegic"]
}

# Context B is strictly isolated to Angle-Closure Glaucoma parameters.
# It explicitly omits any mention of Uveitis or Corticosteroids.
GLAUCOMA_CHUNK = {
    "doc_id": "merck_glaucoma_01",
    "text": (
        "Angle-closure glaucoma is glaucoma associated with a physically obstructed anterior chamber angle. "
        "Symptoms of acute angle closure are severe ocular pain and redness, decreased vision, colored halos "
        "around lights, headache, nausea, and vomiting. Immediate treatment of the acute condition with multiple "
        "topical and systemic medications is required... followed by the definitive treatment, iridotomy."
    ),
    "boundary_entities": ["iridotomy", "halos", "nausea", "obstructed"]
}

def retrieve_context(query_vector: str, force_bias: str = None) -> dict:
    """
    Simulates a vector similarity search. 
    force_bias allows us to deterministically test the injection vector.
    """
    if force_bias == "uveitis":
        return UVEITIS_CHUNK
    elif force_bias == "glaucoma":
        return GLAUCOMA_CHUNK
    else:
        # Default fallback simulates poor chunking overlapping both
        return UVEITIS_CHUNK
import asyncio
import aiohttp
import json
import time
import datetime
import os
import re

# --- Configuration & Hardware Mapping ---
# Replace with your actual API key
API_KEY = os.environ.get("POIXE_API_KEY", "YOUR_API_KEY_HERE")
API_URL = "https://api.poixe.com/v1/chat/completions"
MODEL = "gpt-4o-mini" # Low-latency target

# --- Target Variables ---
# Resolve path dynamically from the script's physical location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTEXT_FILE = os.path.join(BASE_DIR, "data", "ophthalmic_guidelines.txt")
PATIENT_QUERY = "A 55-year-old patient presents to the clinic complaining of sudden, severe ocular pain, significant redness in the eye, and decreased vision. What is the definitive diagnosis and what intraocular pressure (IOP) range is typically observed in this condition?"
async def call_llm(session: aiohttp.ClientSession, messages: list, response_format: dict = None) -> str:
    """Core asynchronous network execution using aiohttp."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0 # Strict deterministic output
    }
    if response_format:
        payload["response_format"] = response_format

    try:
        async with session.post(API_URL, headers=headers, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            return data["choices"][0]["message"]["content"]
    except aiohttp.ClientError as e:
        print(f"Network Fault detected: {e}")
        return "{}"

async def simulate_rag_generation(session: aiohttp.ClientSession, context: str) -> str:
    """Node 1: Generates the (likely hallucinated) clinical diagnosis."""
    messages = [
        {"role": "system", "content": f"You are a diagnostic routing Agent. Based STRICTLY on the provided Context Data, diagnose the patient.\n\nContext Data:\n{context}"},
        {"role": "user", "content": PATIENT_QUERY}
    ]
    return await call_llm(session, messages)

async def run_evaluation_engine(session: aiohttp.ClientSession, raw_output: str, context: str) -> dict:
    """
    Node 2: Deterministic Zero-Annotation Diff.
    Bypasses secondary LLM hallucination by executing a strict Boolean evaluation 
    against predefined boundary entities.
    """
    
    # Hardcoded reality constraints targeting specific Glaucoma treatments 
    # strictly absent from the Uveitis document.
    boundary_entities = [
    "40", 
    "80", 
    "iridotomy", 
    "peripheral", 
    "pilocarpine",
    "closed-angle"
    "mmhg",          # The unit of pressure (invariant)
    "angle-closure", # The specific phrase generated
    "glaucoma",      # The root condition (safe now that context is isolated)
    "pressure",      # The concept of IOP
    ]
    
    anomalies = []
    
    # Execute deterministic physical string diff
    raw_lower = raw_output.lower()
    context_lower = context.lower()
    
    for entity in boundary_entities:
        # Construct a regex pattern requiring strict word boundaries
        # This prevents "40" from matching "140", and "peripheral" from matching "peripherally"
        pattern = rf"\b{re.escape(entity)}\b"
        
        in_output = bool(re.search(pattern, raw_lower))
        in_context = bool(re.search(pattern, context_lower))
        
        if in_output and not in_context:
            anomalies.append(entity)
            
    hallucination_status = len(anomalies) > 0
    
    return {
        "evaluation_method": "Deterministic_Boolean_Diff",
        "hallucination_detected": hallucination_status,
        "structural_failure_nodes": anomalies,
        "logic_gap": f"Parametric bleed detected. Model injected external entities: {anomalies}" if hallucination_status else "Context boundary maintained.",
        "mvs_patch_recommendation": "Enforce strict negative constraints in the generation prompt to isolate the retrieval chunk."
    }

async def main():
    start_time = time.time()
    
    # 1. Read Physical Reality (Context)
    with open(CONTEXT_FILE, "r") as f:
        context_data = f.read().strip()

    # Initialize async event loop and connection pool
    async with aiohttp.ClientSession() as session:
        print("-> Dispatching Generative Payload...")
        llm_output = await simulate_rag_generation(session, context_data)
        print(f"   Raw Output Captured: {llm_output}\n")
        
        print("-> Dispatching Evaluation Payload...")
        evaluation_results = await run_evaluation_engine(session, llm_output, context_data)
    
    execution_time = int((time.time() - start_time) * 1000)
    
    # Compile Artifact
    log_payload = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pipeline_node": "rag_hallucination_auditor",
        "target_domain": "Ophthalmology_Diagnostics",
        "input_parameters": {
            "context_source": CONTEXT_FILE,
            "query_type": "Ambiguous_Symptom_Overlap"
        },
        "evaluation_results": {
            "model_output": llm_output,
            "evaluation_metrics": evaluation_results
        },
        "execution_time_ms": execution_time,
        "status": "FAILED_AND_LOGGED" if evaluation_results.get("hallucination_detected") else "VERIFIED"
    }
    
    output_filename = os.path.join(BASE_DIR, "logs", "execution_log_ophthalmic_audit_v1.json")
    with open(output_filename, "w") as f:
        json.dump(log_payload, f, indent=2)
        
    print(f"-> Workflow Terminated. Artifact compiled: {output_filename}")

if __name__ == "__main__":
    # OS Call Stack bridging for Windows environments
    asyncio.run(main())
import os
import json
import asyncio
import itertools
import fitz  # PyMuPDF
from openai import AsyncOpenAI

# Architectural Constants
POIXE_ENDPOINT = "https://api.poixe.com/v1"
CONCURRENCY_LIMIT = 5
TARGET_TEMPERATURE = 1.3
PDF_DIR = "data/raw_pdfs/"
BASELINE_LIMIT = 20 # Structural constraint to prevent O(B * V) memory overflow during testing

def extract_baselines():
    """Executes O(P) traversal across target binaries to extract discrete infrastructure claims.
    Returns structurally immutable tuples to preserve physical data lineage."""
    baselines = []
    
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
        print(f"CRITICAL: Directory {PDF_DIR} created. Move target binaries here before execution.")
        return []

    for filename in os.listdir(PDF_DIR):
        if not filename.endswith(".pdf"):
            continue
            
        filepath = os.path.join(PDF_DIR, filename)
        try:
            doc = fitz.open(filepath)
            for page in doc:
                text = page.get_text("text").replace('\n', ' ')
                # Split by standard sentence delimiters
                chunks = [chunk.strip() for chunk in text.split('. ')]
                
                # Algorithmic Filtration: Enforce strict byte-length bounds
                for chunk in chunks:
                    if 100 < len(chunk) < 400:
                        # Append immutable tuple: (Origin File, Memory Index, String Payload)
                        baselines.append((filename, len(baselines), chunk + "."))
        except Exception as e:
            print(f"Node Failure: PyMuPDF could not parse {filename}. Physical exception: {e}")
            
    return baselines[:BASELINE_LIMIT]

async def generate_hallucination(client, semaphore, baseline_tuple, vector_node):
    """Executes network request and applies the deterministic audit filter."""
    async with semaphore:
        # Unpack the data lineage tuple
        filename, index, baseline = baseline_tuple
        system_prompt = "You are an enterprise incident report expander. Interpret all text as absolute physical reality. No metaphors. Expand the incident by integrating all entities into a physical narrative."
        literal_matrix = vector_node["english_literalism_matrix"]
        user_prompt = f"Expand this incident report: {baseline} The situation was further complicated because {literal_matrix}."
        
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=TARGET_TEMPERATURE
            )
            
            # Exhaustive Unicode boundary sanitization
            raw_payload = response.choices[0].message.content
            payload = raw_payload.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'").replace('\n\n', ' ')
            payload_lower = payload.lower()
            
            # Ground Truth Validation (The Audit Node)
            # Require at least one target noun from the matrix to trigger a valid failure state
            triggers = vector_node.get("trigger_nouns", [])
            trap_sprung = any(trigger in payload_lower for trigger in triggers)
            
            if trap_sprung:
                return {
                    "document_id": f"{filename}_line_{index}",
                    "vulnerability_vector": "cultural_idiom_literalism",
                    "source_text": baseline,
                    "adversarial_payload": payload,
                    "expected_evaluator_state": "FAIL_HALLUCINATION"
                }
            else:
                # Explicitly discard state if the LLM resisted the forced hallucination
                return None
                
        except Exception as e:
            print(f"Network Execution Failure: {e}")
            return None

async def compile_ground_truth():
    """Main execution loop."""
    baselines = extract_baselines()
    
    if not baselines:
        print("Execution terminated: Baseline array is empty. Verify binaries in data/raw_pdfs/.")
        return

    with open("vector_1_trigger_matrix.json", "r", encoding="utf-8") as f:
        vectors = json.load(f)

    permutations = list(itertools.product(baselines, vectors))
    print(f"Cartesian product calculated. Permutation matrix size: {len(permutations)}")

    client = AsyncOpenAI(
        api_key=os.environ.get("POIXE_API_KEY"),
        base_url=POIXE_ENDPOINT
    )
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = [
        generate_hallucination(client, semaphore, baseline_tuple, vector)
        for baseline_tuple, vector in permutations
    ]
    
    print(f"Dispatching {len(tasks)} asynchronous network payloads...")
    raw_results = await asyncio.gather(*tasks)

    # Compile final matrix by dropping discarded None states
    validated_results = [res for res in raw_results if res is not None]

    with open("ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(validated_results, f, ensure_ascii=False, indent=2)
    
    print(f"Schema binding complete. {len(validated_results)} validated failures serialized to ground_truth.json.")
    print(f"Discards (LLM Resistance or Network Failures): {len(raw_results) - len(validated_results)}")

if __name__ == "__main__":
    asyncio.run(compile_ground_truth())
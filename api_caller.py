import asyncio
import json
import os
import aiohttp
from typing import Set

# Strict environment isolation
API_KEY = os.getenv("LLM_API_KEY")
if not API_KEY:
    raise ValueError("CRITICAL: LLM_API_KEY not found in OS environment.")

INPUT_FILE = "ground_truth.json"
OUTPUT_FILE = "output.jsonl"
MAX_CONCURRENCY = 5
MAX_RETRIES = 3

def load_completed_nodes() -> Set[str]:
    """Idempotency check: Load already processed node_ids to prevent duplicate execution."""
    completed = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    completed.add(data.get("node_id"))
    return completed

async def fetch_llm_response(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, payload: dict) -> dict:
    """Network execution with semaphore throttling and exponential backoff."""
    node_id = payload["node_id"]
    prompt = payload["input_prompt"]
    
    # Target endpoint routing
    url = "https://api.openai.com/v1/chat/completions" # Replace with target LLM endpoint
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "gpt-4o-mini", # Replace with target model
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0 # Strict zero variance required for baseline deterministic evaluation
    }

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(url, headers=headers, json=body) as response:
                    # Intercept rate limits and server drops
                    if response.status in [429, 503, 502, 504]:
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                        continue
                    
                    response.raise_for_status()
                    data = await response.json()
                    raw_text = data["choices"][0]["message"]["content"]
                    
                    return {"node_id": node_id, "raw_text": raw_text}
            
            except Exception as e:
                # Hard fail capture after max retries
                if attempt == MAX_RETRIES - 1:
                    return {"node_id": node_id, "error": str(e)}
                await asyncio.sleep(2 ** attempt)

async def main():
    completed_nodes = load_completed_nodes()
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
        
    target_nodes = [node for node in ground_truth if node["node_id"] not in completed_nodes]
    
    if not target_nodes:
        print("Execution Halted: Zero target nodes in queue. All nodes processed.")
        return

    # Initialize throttling gate
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    
    # Persistent session prevents localized socket exhaustion
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_llm_response(session, semaphore, node) for node in target_nodes]
        
        for completed_task in asyncio.as_completed(tasks):
            result = await completed_task
            
            # Atomic append strictly locks state to local disk immediately
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_f:
                out_f.write(json.dumps(result, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    asyncio.run(main())

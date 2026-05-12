import os
import re
import csv
import asyncio
import aiohttp
import json

# Structural Migration: Domestic Relay Node
API_KEY = os.environ.get("POIXE_API_KEY", "").strip()
API_URL = "https://api.poixe.com/v1/chat/completions"

# Explicit Free-Tier Ledger Routing
MODEL = "gpt-4o-mini" 
MAX_CONCURRENCY = 5

async def evaluate_claim(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, ground_truth: str, claim: str) -> dict:
    """Executes deterministic Boolean verification against the ground truth matrix via Poixe Relay."""
    
    if not API_KEY:
        return {"claim": claim, "status": "EXECUTION_FAILURE", "reason": "POIXE_API_KEY is NULL."}
        
    system_prompt = (
        "You are a strict, deterministic evaluation matrix. "
        "Your sole function is to verify if the TARGET CLAIM physically exists within the SOURCE DOCUMENT. "
        "If the claim is supported, output strictly: PASS | [1-sentence physical explanation]. "
        "If the claim contains unverified entities, metrics, or logic not in the source, output strictly: FAIL_HALLUCINATION | [1-sentence physical explanation]. "
        "Do not output any other text."
    )
    
    user_prompt = f"SOURCE DOCUMENT:\n{ground_truth}\n\nTARGET CLAIM:\n{claim}"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0
    }
    
    async with semaphore:
        try:
            # Explicit physical socket routing restored for OS VPN compatibility
            async with session.post(API_URL, headers=headers, json=payload, proxy="http://127.0.0.1:7890") as response:
                if response.status != 200:
                    return {"claim": claim, "status": "API_ERROR", "reason": await response.text()}
                
                data = await response.json()
                raw_output = data['choices'][0]['message']['content'].strip()
                
                parts = raw_output.split("|", 1)
                status = parts[0].strip()
                reason = parts[1].strip() if len(parts) > 1 else "Parsing Failure"
                
                return {"claim": claim, "status": status, "reason": reason}
        except Exception as e:
            return {"claim": claim, "status": "EXECUTION_FAILURE", "reason": str(e)}

def extract_claims(markdown_path: str) -> list:
    """Isolates discrete bullet points from the generative payload."""
    claims = []
    with open(markdown_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
        matches = re.findall(r'-\s*\*\*(.*?)\*\*(.*)', content)
        for match in matches:
            claims.append(f"{match[0]}: {match[1].strip()}")
    return claims

async def execute_evaluation_pipeline():
    truth_path = "mac_mini_ground_truth.txt"
    payload_path = "chat.md"
    output_csv = "live_diff_sheet.csv"
    
    if not os.path.exists(truth_path) or not os.path.exists(payload_path):
         raise FileNotFoundError("CRITICAL: Input files missing. Establish mac_mini_ground_truth.txt and chat.md.")
         
    with open(truth_path, 'r', encoding='utf-8', errors='replace') as f:
        ground_truth_text = f.read()
        
    claims = extract_claims(payload_path)
    print(f"Pipeline Initialized: Extracted {len(claims)} discrete logic nodes for verification.")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    
    async with aiohttp.ClientSession() as session:
        tasks = [evaluate_claim(session, semaphore, ground_truth_text, claim) for claim in claims]
        results = await asyncio.gather(*tasks)
        
    # Write output to CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Target Claim", "Eval Status", "Architectural Truth"])
        for r in results:
            writer.writerow([r['claim'], r['status'], r['reason']])
            
    print(f"Execution Complete: Verification matrix compiled to {output_csv}.")

if __name__ == "__main__":
    asyncio.run(execute_evaluation_pipeline())
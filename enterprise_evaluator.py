import os
import re
import csv
import asyncio
import aiohttp
import json
import fitz  # PyMuPDF

# Structural Mapping: Domestic Relay Node
API_KEY = os.environ.get("POIXE_API_KEY", "").strip()
API_URL = "https://api.poixe.com/v1/chat/completions"
MODEL = "gpt-4o-mini:free" 
MAX_CONCURRENCY = 5

async def evaluate_claim(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, ground_truth: str, claim: str) -> dict:
    """Executes deterministic Boolean verification with Strict JSON Enforcement."""
    
    if not API_KEY:
        return {"claim": claim, "status": "EXECUTION_FAILURE", "reason": "POIXE_API_KEY is NULL."}
        
    system_prompt = (
        "You are a strict, deterministic evaluation matrix. "
        "Verify if the TARGET CLAIM physically exists within the SOURCE DOCUMENT. "
        "Output strictly matching the required JSON schema."
    )
    
    user_prompt = f"SOURCE DOCUMENT:\n{ground_truth}\n\nTARGET CLAIM:\n{claim}"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Architectural Shift: Strict JSON Schema Definition
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "hallucination_verification_matrix",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["PASS", "FAIL_HALLUCINATION"],
                            "description": "Deterministic boolean evaluation state."
                        },
                        "reason": {
                            "type": "string",
                            "description": "A strict 1-sentence physical explanation."
                        }
                    },
                    "required": ["status", "reason"],
                    "additionalProperties": False
                }
            }
        }
    }
    
    async with semaphore:
        try:
            # Explicit physical socket routing restored
            async with session.post(API_URL, headers=headers, json=payload, proxy="http://127.0.0.1:7890") as response:
                if response.status != 200:
                    return {"claim": claim, "status": "API_ERROR", "reason": await response.text()}
                
                data = await response.json()
                
                # The output is now mathematically guaranteed to be a valid JSON string
                raw_json_string = data['choices'][0]['message']['content']
                parsed_data = json.loads(raw_json_string)
                
                return {
                    "claim": claim, 
                    "status": parsed_data["status"], 
                    "reason": parsed_data["reason"]
                }
        except Exception as e:
            return {"claim": claim, "status": "EXECUTION_FAILURE", "reason": str(e)}

def ingest_enterprise_document(file_path: str) -> str:
    """Executes C-level text extraction based on binary file signature."""
    print(f"Ingesting binary node: {file_path}")
    if file_path.lower().endswith('.pdf'):
        try:
            doc = fitz.open(file_path)
            text = chr(10).join([page.get_text() for page in doc])
            doc.close()
            return text
        except Exception as e:
             raise RuntimeError(f"Binary PDF extraction failed: {str(e)}")
    else:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()

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
    # You may now target physical PDF files directly
    truth_path = "Mac-mini-(M1,-2020)-Service-Guide.pdf" 
    payload_path = "chat.md"
    output_csv = "enterprise_diff_sheet.csv"
    
    if not os.path.exists(truth_path) or not os.path.exists(payload_path):
         raise FileNotFoundError(f"CRITICAL: Input files missing. Establish {truth_path} and {payload_path}.")
         
    ground_truth_text = ingest_enterprise_document(truth_path)
    claims = extract_claims(payload_path)
    print(f"Pipeline Initialized: Extracted {len(claims)} discrete logic nodes for verification.")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    
    async with aiohttp.ClientSession() as session:
        tasks = [evaluate_claim(session, semaphore, ground_truth_text, claim) for claim in claims]
        results = await asyncio.gather(*tasks)
        
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Target Claim", "Eval Status", "Architectural Truth"])
        for r in results:
            writer.writerow([r['claim'], r['status'], r['reason']])
            
    print(f"Execution Complete: Strict JSON verification matrix compiled to {output_csv}.")

if __name__ == "__main__":
    asyncio.run(execute_evaluation_pipeline())
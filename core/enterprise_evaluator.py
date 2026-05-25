import os
import re
import csv
import asyncio
import aiohttp
import json
import fitz  # PyMuPDF
import argparse
import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Structural Mapping: Domestic Relay Node
API_KEY = os.environ.get("POIXE_API_KEY", "").strip()
API_URL = "https://api.poixe.com/v1/chat/completions"
MODEL = "gpt-4o-mini:free" 
MAX_CONCURRENCY = 5

def ingest_enterprise_document(file_path: str) -> list:
    """Executes C-level binary text extraction, returning an array of discrete chunks (pages)."""
    print(f"Ingesting binary node for semantic chunking: {file_path}")
    chunks = []
    if file_path.lower().endswith('.pdf'):
        try:
            doc = fitz.open(file_path)
            # Architectural Shift: Isolate each page as a discrete memory block
            chunks = [page.get_text().strip() for page in doc if page.get_text().strip()]
            doc.close()
            return chunks
        except Exception as e:
             raise RuntimeError(f"Binary PDF extraction failed: {str(e)}")
    else:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            # Fallback for txt: split roughly by double line breaks
            text = f.read()
            chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
            return chunks

def retrieve_relevant_context(chunks: list, claim: str, top_k: int = 2) -> str:
    """Executes TF-IDF vectorization to calculate cosine similarity and isolate relevant context."""
    if not chunks:
        return ""
        
    # Translate strings into a sparse mathematical matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(chunks + [claim])
    
    # Calculate angular distance between the claim (last item) and all document chunks
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Extract the array indices of the highest scoring chunks
    top_indices = cosine_similarities.argsort()[-top_k:][::-1]
    
    # Reconstruct the optimized context block
    targeted_context = "\n---[CONTEXT GAP]---\n".join([chunks[i] for i in top_indices])
    return targeted_context

async def evaluate_claim(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, optimized_context: str, claim: str) -> dict:
    """Executes deterministic Boolean verification using targeted semantic context."""
    
    if not API_KEY:
        return {"claim": claim, "status": "EXECUTION_FAILURE", "reason": "POIXE_API_KEY is NULL."}
        
    system_prompt = (
        "You are a strict, deterministic evaluation matrix. "
        "Verify if the TARGET CLAIM physically exists within the SOURCE DOCUMENT. "
        "Output strictly matching the required JSON schema."
    )
    
    # The prompt now receives mathematically isolated context, preventing window saturation
    user_prompt = f"SOURCE DOCUMENT:\n{optimized_context}\n\nTARGET CLAIM:\n{claim}"
    
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
            async with session.post(API_URL, headers=headers, json=payload, proxy="http://127.0.0.1:7890") as response:
                if response.status != 200:
                    return {"claim": claim, "status": "API_ERROR", "reason": await response.text()}
                
                data = await response.json()
                raw_json_string = data['choices'][0]['message']['content']
                parsed_data = json.loads(raw_json_string)
                
                return {
                    "claim": claim, 
                    "status": parsed_data["status"], 
                    "reason": parsed_data["reason"]
                }
        except Exception as e:
            return {"claim": claim, "status": "EXECUTION_FAILURE", "reason": str(e)}

def extract_claims(file_path: str) -> list:
    """Parses a generative markdown payload and extracts discrete claims as string elements."""
    claims = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                # Filter out empty strings and structural markdown metadata
                if stripped_line and not stripped_line.startswith(('#', '---', '***')):
                    # Strip leading bullet/list characters to isolate the raw semantic claim
                    clean_claim = re.sub(r'^[\-\*\+]\s+', '', stripped_line)
                    claims.append(clean_claim)
    except Exception as e:
        print(f"I/O Error reading payload node {file_path}: {str(e)}")
    return claims

async def process_batch(truth_path: str, payload_path: str, output_csv: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore):
    """Executes the pipeline using TF-IDF routing."""
    try:
        # Ingestion now returns a list of pages
        ground_truth_chunks = ingest_enterprise_document(truth_path)
        claims = extract_claims(payload_path)
        print(f"[{os.path.basename(payload_path)}] Extracted {len(claims)} nodes.")
        
        if not claims:
            return
            
        tasks = []
        for claim in claims:
            # Architectural Shift: Calculate semantic similarity before hitting the network
            optimized_context = retrieve_relevant_context(ground_truth_chunks, claim, top_k=2)
            tasks.append(evaluate_claim(session, semaphore, optimized_context, claim))
            
        results = await asyncio.gather(*tasks)
        
        with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for r in results:
                writer.writerow([os.path.basename(payload_path), r['claim'], r['status'], r['reason']])
    except Exception as e:
        print(f"CRITICAL FAILURE on {payload_path}: {str(e)}")

async def scale_evaluation_pipeline(truth_dir: str, payload_dir: str, output_csv: str):
    """Dynamic directory routing for mass evaluation."""
    print(f"Initializing V3 Scalable Pipeline on targets: {payload_dir} -> {truth_dir}")
    
    payload_files = glob.glob(os.path.join(payload_dir, "*.md"))
    if not payload_files:
        raise FileNotFoundError("No markdown payloads found in the target directory.")

    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Source Payload", "Target Claim", "Eval Status", "Architectural Truth"])

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    
    async with aiohttp.ClientSession() as session:
        # MVS Assumes one master PDF for the batch. Can be scaled dynamically later.
        truth_path = os.path.join(truth_dir, "Mac-mini-(M1,-2020)-Service-Guide.pdf") 
        
        for payload_path in payload_files:
            await process_batch(truth_path, payload_path, output_csv, session, semaphore)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic LLM Hallucination Evaluation Engine")
    parser.add_argument("--truth_dir", type=str, required=True, help="Directory containing enterprise PDF ground truth data.")
    parser.add_argument("--payload_dir", type=str, required=True, help="Directory containing generative markdown payloads.")
    parser.add_argument("--output", type=str, default="master_evaluation.csv", help="Target CSV output matrix.")
    
    args = parser.parse_args()
    
    asyncio.run(scale_evaluation_pipeline(args.truth_dir, args.payload_dir, args.output))
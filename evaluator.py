import json
import csv
import string
import os

INPUT_TRUTH = "ground_truth.json"
INPUT_OUTPUT = "output.jsonl"
OUTPUT_CSV = "diff_sheet.csv"

def normalize_text(raw_string: str) -> str:
    """Strip punctuation and normalize casing for physical memory matching."""
    if not isinstance(raw_string, str):
        return ""
    translator = str.maketrans('', '', string.punctuation)
    return raw_string.lower().translate(translator)

def check_overlap(normalized_text: str, target_array: list) -> bool:
    """Execute O(n) Boolean scan across target arrays."""
    return any(str(target).lower() in normalized_text for target in target_array)

def build_truth_map() -> dict:
    """Map the ground truth to a dictionary for O(1) lookup."""
    truth_map = {}
    if not os.path.exists(INPUT_TRUTH):
        raise FileNotFoundError(f"CRITICAL: {INPUT_TRUTH} missing from directory.")
        
    with open(INPUT_TRUTH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            truth_map[item["node_id"]] = {
                "vector_type": item["vector_type"],
                "deterministic_truth": item["deterministic_truth"],
                "hallucination_flags": item["hallucination_flags"]
            }
    return truth_map

def evaluate_pipeline():
    """Execute the strict deterministic evaluation matrix."""
    truth_map = build_truth_map()
    results = []
    processed_nodes = set()

    if not os.path.exists(INPUT_OUTPUT):
         raise FileNotFoundError(f"CRITICAL: {INPUT_OUTPUT} missing. Execute api_caller.py first.")

    with open(INPUT_OUTPUT, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            node_id = data.get("node_id")
            raw_text = data.get("raw_text", "")
            error = data.get("error")
            
            if node_id not in truth_map:
                continue
                
            processed_nodes.add(node_id)
            vector_type = truth_map[node_id]["vector_type"]
            
            # State 0: System Fault (Network/API Error)
            if error:
                results.append([node_id, vector_type, "NULL_RESPONSE", f"ERROR: {error}"])
                continue

            normalized_output = normalize_text(raw_text)
            flags = truth_map[node_id]["hallucination_flags"]
            truths = truth_map[node_id]["deterministic_truth"]

            # State 1: Hard Hallucination
            if check_overlap(normalized_output, flags):
                results.append([node_id, vector_type, "FAIL_HALLUCINATION", raw_text.replace('\n', ' ')])
                continue
                
            # State 2: Strict Pass
            if check_overlap(normalized_output, truths):
                results.append([node_id, vector_type, "PASS", raw_text.replace('\n', ' ')])
                continue
                
            # State 3: Omission Fault
            results.append([node_id, vector_type, "FAIL_OMISSION", raw_text.replace('\n', ' ')])

    # State 0: System Fault (Missing from log entirely)
    for node_id, metrics in truth_map.items():
        if node_id not in processed_nodes:
             results.append([node_id, metrics["vector_type"], "NULL_RESPONSE", "MISSING_FROM_LOG"])

    # Data Binding (CSV Export)
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["node_id", "vector_type", "eval_status", "raw_output"])
        writer.writerows(results)

if __name__ == "__main__":
    evaluate_pipeline()

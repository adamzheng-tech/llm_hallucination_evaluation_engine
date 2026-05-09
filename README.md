# LLM Hallucination Evaluation Engine

An asynchronous, deterministic evaluation pipeline designed to quantify LLM structural failures against cross-domain linguistic and syntactic arbitrage vectors.

## 1. System Architecture

* **`api_caller.py` (I/O Execution):** Asynchronous network engine utilizing semaphore throttling to bypass latency constraints while maintaining state idempotency via atomic `.jsonl` appends.
* **`ground_truth.json` (State Reality):** Proprietary array of structured logical traps, exploiting cultural idiom literalism, cross-domain semantic collisions, and strict physical reality parameters.
* **`evaluator.py` (Boolean Verification):** Deterministic scoring matrix executing normalized string matching against raw API payloads to calculate structural adherence to reality.

## 2. Execution Protocol

**CRITICAL:** Absolute environment isolation is required. Never execute without a `.env` file containing the target cryptographic token or injecting it directly into the active OS session.

    pip install aiohttp
    set LLM_API_KEY=your_target_key
    python api_caller.py
    python evaluator.py

## 3. Evaluated Vulnerability Vectors

| Vector ID | Foundational Logic Trap | Arbitrage Domain |
| :--- | :--- | :--- |
| `cultural_idiom_literalism` | Evaluates LLM reliance on literal translation over contextual anatomical reality. | High-Context Business Pragmatics |
| `geo_academic_slang` | Tests failure rates against highly localized socioeconomic involution terminology. | Regional Academic Lexicons |
| `cross_domain_collision` | Calculates structural divergence when identical syntax occupies distinct physical engineering and linguistic realities. | CFD vs. TCSOL Semantics |
| `high_context_omission` | Forces strict zero-inference neutrality in gender-omitted syntax arrays. | Linguistic Structural Constraints |

## 4. Time & Space Complexity

* **Network Execution:** O(N / M) where N is the dataset node count and M is the strictly enforced asynchronous concurrency limit (M=5).
* **Evaluation Engine:** O(1) memory lookup via the truth_map dictionary, combined with an O(N * T) Boolean scan, where T is the normalized target array length.
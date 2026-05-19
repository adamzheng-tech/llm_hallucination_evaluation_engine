# LLM Hallucination Evaluation Engine

> **RESTRICTED DEPLOYMENT LICENSE (Proprietary / CC BY-NC 4.0)**
> *The deterministic evaluation engine, testing matrices, and vulnerability payloads contained within this repository are proprietary intellectual property. Commercial deployment, integration into enterprise CI/CD pipelines, or usage by corporate entities for internal Large Language Model auditing strictly requires commercial authorization. Academic and local non-commercial review is permitted.*

A deterministic Retrieval-Augmented Generation (RAG) evaluation pipeline engineered to cross-reference generative payloads against unstructured enterprise documentation.

## 1. Production Architecture (V2)

* **Dynamic Data Ingestion:** Utilizes `PyMuPDF` for C-level binary text extraction from native PDF assets (e.g., technical service guides, enterprise earnings reports), bypassing fragile plaintext dependencies.
* **Domestic API Relay:** Routes asynchronous HTTP payloads through the Poixe API gateway (`gpt-4o-mini:free`), eliminating international DNS friction while preserving standard REST schemas.
* **Deterministic Evaluation Matrix:** Enforces Strict JSON Schema output (`response_format`) to mathematically bind the LLM-as-a-Judge to a binary Boolean state (`PASS` or `FAIL_HALLUCINATION`), stripping conversational autonomy and preventing output parsing failures.

## 2. Execution Protocol

CRITICAL: Absolute environment isolation is required. Never execute without injecting the target cryptographic token directly into the active OS session.

```bash
pip install aiohttp PyMuPDF
set POIXE_API_KEY=your_target_key
python enterprise_evaluator.py
```

## 3. Evaluated Vulnerability Vectors

The following table:

| Vector ID | Foundational Logic Trap | Arbitrage Domain |
| :--- | :--- | :--- |
| `cultural_idiom_literalism` | Evaluates LLM reliance on literal translation over contextual anatomical reality. | High-Context Business Pragmatics |
| `geo_academic_slang` | Tests failure rates against highly localized socioeconomic involution terminology. | Regional Academic Lexicons |
| `cross_domain_collision` | Calculates structural divergence when identical syntax occupies distinct physical engineering and linguistic realities. | CFD vs. TCSOL Semantics |
| `high_context_omission` | Forces strict zero-inference neutrality in gender-omitted syntax arrays. | Linguistic Structural Constraints |

## 4. Algorithmic Complexity

* **Ingestion Phase:** $O(P)$ time complexity, where $P$ is the physical PDF page count.
* **Network Execution:** $O(C/M)$ wall-clock duration, where $C$ is the total discrete claim count and $M$ is the strictly enforced aiohttp concurrency limit ($M=5$).
* **Evaluation Engine:** $O(1)$ memory lookup combined with an $O(N \cdot T)$ Boolean scan where $T$ is the normalizer.
* **Space Complexity:** $O(S)$ linear memory allocation strictly bound to the extracted document string length during execution.
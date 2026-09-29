# LLM Hallucination Evaluation Engine

A Python learning and portfolio project for checking claims against a supplied document.
It retrieves passages with TF-IDF, asks an LLM to assess support, validates quoted evidence,
and writes an auditable CSV. Results are model judgments, not guarantees of truth.

## Quick start (macOS / Linux, Python 3.10+)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python core/enterprise_evaluator.py --source examples/reference.txt --claims examples/claims.md --dry-run --output outputs/preview.csv
python -m unittest discover -s tests -v
```

Run commands from the repository root. Use a new output filename on each run; existing
outputs are not overwritten. `--dry-run` performs real document extraction and retrieval,
but makes **no API calls** and labels every row `NOT_EVALUATED`.

## Live evaluation

Configure an OpenAI-compatible Chat Completions provider and a model available to your account.
Do not commit keys. `.env.example` documents names; this program does **not** auto-load `.env`.
In macOS's default zsh, enter the key without displaying it or storing it in shell history:

```zsh
read -s 'OPENAI_API_KEY?API key: '; echo
export OPENAI_API_KEY
export OPENAI_BASE_URL='https://your-provider.example/v1'
export OPENAI_MODEL_NAME='your-supported-model'
python core/enterprise_evaluator.py --source examples/reference.txt --claims examples/claims.md --output outputs/live.csv
```

`POIXE_API_KEY` is also accepted as a fallback. Model and provider have no hardcoded defaults.
No proxy is used unless you explicitly pass `--proxy URL`.
Default response format is `json_object`; use `--response-format json_schema` if your provider
supports strict structured outputs, or `--response-format text` if it supports neither.
All modes validate the returned object and quotes locally. There is no automatic format downgrade.

Options: `--top-k 2`, `--timeout 60` (seconds per attempt), `--concurrency 3`, `--attempts 3`.
Only connection failures, timeouts, HTTP 429 and HTTP 5xx are retried, with bounded exponential delay.
Malformed responses and permanent HTTP errors become `ERROR` rows. Exit codes: 0 = completed,
1 = one or more claim errors, 2 = input/configuration failure. A completed run does not imply accurate judgments.

## Data flow and output

1. Read one PDF (text layer), TXT or Markdown source; keep page/paragraph identifiers.
2. Read one claim per line from the claims file; ignore headings and separators.
3. Fit TF-IDF once on the document and retrieve passages per claim.
4. Ask the model for a judgment with exact quotes and validate those quotes against retrieved text.
5. Save claim, source path, requested model, status, reason, evidence and full retrieved passages to CSV.

| Status | Meaning |
| --- | --- |
| `SUPPORTED` | Retrieved passages support the entire claim |
| `CONTRADICTED` | Retrieved passages explicitly conflict with the claim |
| `INSUFFICIENT_EVIDENCE` | Retrieved passages do not settle the claim |
| `NOT_EVALUATED` | Retrieval preview only |
| `ERROR` | Request or output validation failed |

`SUPPORTED` and `CONTRADICTED` require nonempty verbatim citations. Quote validation proves
that the text occurs in a retrieved passage; it does not prove that the quote justifies the verdict.

## Verification and limitations

- `examples/expected.json` contains manually specified labels for three fictional examples.
  Compare these with a live run; these examples are a smoke check, not an accuracy benchmark.
- Automated tests cover retrieval, PDF page IDs, quote validation, dry-run, CSV transport,
  transient retries and permanent failures using a local mock API. They do not measure LLM accuracy.
- Lexical retrieval can miss synonyms, negation context and evidence on other pages. Absence
  from retrieved passages does not establish falsity. A zero-overlap result is marked insufficient.
- English stop words are used. Chinese and multilingual retrieval have not been evaluated.
- Scanned PDFs need OCR; tables and reading order can be imperfect. Long pages may exceed model limits.
- One line is treated as one claim; automatic atomic-claim extraction is not implemented.
- Schema constraints and low temperature cannot guarantee semantic correctness or determinism.
- Prompt instructions reduce but do not eliminate prompt-injection risk.
- No held-out accuracy, latency or cost benchmark has been established. This is not a production or clinical system.

## Repository map

- `core/enterprise_evaluator.py`: maintained CLI and main learning path.
- `tests/`: offline regression tests.
- `examples/`: small fictional reference, claims and expected labels.
- [中文学习路线](docs/LEARNING.md): how to understand and extend the main path.
- Other `core/` scripts, existing `data/`, `claims/`, `logs/` and `master_evaluation.csv`:
  historical experiments and artifacts, outside the maintained CLI. Historical outputs are not
  validation results for this revision. Some scripts use different paths/providers.

The former `--truth_dir` / `--payload_dir` interface is replaced by explicit `--source` / `--claims`
file arguments. This avoids silently evaluating every payload against one hardcoded PDF.

## License

See [LICENSE](LICENSE) for the repository's existing noncommercial terms. Third-party source
documents retain their respective rights; the fictional quick-start examples need none of those PDFs.

# my_nlp_projects

This repository contains two related projects for a Question-Answering (QA) agent assignment:
- `baseline_qa_agent`: a baseline implementation
- `qa_agent`: an agent that integrates a required RAG pipeline (`rag_system`) to improve answer grounding

The guide below explains how to generate the dataset, set up environments, run the baselines and the RAG-enabled agent locally, and evaluate results.

## Prerequisites
Run `pip install -r requirements.txt` to download the requirements.

## Repository Structure
- `baseline_qa_agent/HW4_baseline.ipynb`: baseline notebook
- `qa_agent/HW4_baseline.ipynb`: RAG-enabled agent notebook that uses `rag_system` tools
- `countries_with_languages.tsv`: country → official languages mapping
- `public.jsonl` / `private.jsonl`: input datasets (see “Generate the dataset”)
- `rag_system/`: required RAG pipeline used by `qa_agent` (indexing + retrieval tools)

## Environment Setup
Create and activate a Python environment, then install dependencies.

### Option A: venv
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: conda
```bash
conda create -n nlpproj python=3.12 -y
conda activate nlpproj
pip install --upgrade pip
pip install -r requirements.txt
```

If a `requirements.txt` is not present, install the common set used by the notebooks:
```bash
pip install langchain==0.3.3 langchain-core langchain-community faiss-cpu kagglehub \
            langchain-openai langchain-huggingface transformers sentencepiece \
            sentence-transformers bitsandbytes accelerate jinja2 tqdm pandas \
            langdetect
```

Optional: set API keys (only if using those providers)
```bash
export OPENAI_API_KEY=...            # if using LiteLLM/OpenAI
export HUGGINGFACEHUB_API_TOKEN=...  # if pulling gated models
```

## Generate the Dataset
The projects expect JSONL files with one JSON object per line.

- `public.jsonl` and `private.jsonl` follow this schema:
  - GlobalTrekker: `{ "type": "GlobalTrekker", "paragraph": "...", "country": "...", "city": "..." }`
  - CulinaryDetective: `{ "type": "CulinaryDetective", "ingredient": "...", "description": "...", "country": "...", "region": "..." }`
  - LinguaLocale: `{ "type": "LinguaLocale", "sentence": "...", "country": "..." }`

To create your own:
1. Prepare source CSVs or text files with fields above.
2. Convert to JSONL. Example script:
```python
import json, pandas as pd

# Example: build a mixed dataset
data = [
  {"type":"GlobalTrekker","paragraph":"A red double-decker bus near Big Ben.","country":"United Kingdom","city":"London"},
  {"type":"CulinaryDetective","ingredient":"rice,coconut","description":"Steamed cakes with coconut chutney.","country":"India","region":"South"},
  {"type":"LinguaLocale","sentence":"Please, colour is the preferred spelling in our centre.","country":"United Kingdom"}
]
with open("public.jsonl","w") as f:
  for row in data:
    f.write(json.dumps(row)+"\n")
```
3. Place files at repo root as `public.jsonl` and `private.jsonl`.

## Running baseline_qa_agent
Open `baseline_qa_agent/HW4_baseline.ipynb` in Jupyter/VS Code and run cells top-to-bottom.

- Model choice:
  - OpenAI via LiteLLM: set `OPENAI_API_KEY` and ensure `base_url` matches your gateway.
  - Hugging Face: set `HUGGINGFACEHUB_API_TOKEN` and choose a model (e.g., `Qwen/Qwen3-0.6B`).
- GPU: the HF pipeline uses `device_map="auto"` and selects CUDA if available.

Steps inside the notebook:
1. Load dependencies and `countries_with_languages.tsv`.
2. Define message templates and extractors for the three tasks.
3. Load `public.jsonl` (or `private.jsonl`).
4. Generate predictions with tqdm and save to `public.txt` (or `private.txt`) in the format: `type\tcountry\tplace`.
5. Evaluate using the provided scoring cell.

CLI alternative to run the prediction loop from the notebook environment:
```bash
# In Jupyter cell or Python shell with the notebook context loaded
from tqdm import tqdm
answers = []
for q in tqdm(questions):
    country, category = geoguesser(q)
    answers.append(f"{q['type']}\t{country}\t{category}")
with open("public.txt","w") as f:
    for a in answers: f.write(a+"\n")
```

## Running qa_agent (RAG-enabled)
Open `qa_agent/HW4_baseline.ipynb` and run cells top-to-bottom. This agent depends on the `rag_system` package to build indexes and expose retrieval tools used by the CulinaryDetective (and optionally other tasks).

Key components:
- `rag_system/rag_pipeline.py`: constructs a FAISS index, persists/load embeddings, and exposes a `CulinaryRAG` interface.
- `qa_agent/HW4_baseline.ipynb`: imports `CulinaryRAG`, loads the index, and registers tools such as `retrieve_culinary_context` for the agent.

Typical flow in the notebook:
1. Ensure embeddings and FAISS index exist (the notebook calls into `CulinaryRAG` to create or load them).
2. Load `public.jsonl` (or `private.jsonl`).
3. For CulinaryDetective questions, the agent uses the `retrieve_culinary_context` tool to fetch relevant passages before answering.
4. Generate predictions with tqdm and save to `public.txt` (or `private.txt`) in the format `type\tcountry\tplace`.
5. Evaluate using the provided scoring cell (same format as baseline).

If you need to (re)build the RAG index from a Kaggle dataset:
```python
from rag_system.rag_pipeline import CulinaryRAG
rag = CulinaryRAG()
retriever = rag.build_or_load_index()  # builds if missing, else loads existing index
```

## Evaluation Format
The evaluation prints:
```
GlobalTrekker Average Score: X.XXXX
CulinaryDetective Average Score: Y.YYYY
LinguaLocale Average Score: Z.ZZZZ
```
Ensure your predictions files are aligned:
- Baseline: `public.txt` / `private.txt`
- RAG agent: also writes `public.txt` / `private.txt` unless you customize filenames

## Troubleshooting
- IndexError when evaluating: Ensure each line has three tab-separated fields. Pad/truncate when loading.
- Azure/OpenAI content policy errors: sanitize prompts, reduce sensitive terms, or use safer system messages.
- GPU not used: verify `torch.cuda.is_available()` and drivers; for HF, ensure `accelerate`/`bitsandbytes` installed and `device_map="auto"`.
- Missing dependencies: re-run `pip install` commands above.

## Repro Summary
- Create environment and install dependencies.
- Place `public.jsonl` and `private.jsonl` at repository root.
- Run baseline or ReAct notebook:
  - Baseline → produce `public.txt` and evaluate.
  - ReAct → `run_react_on_split("public")` → `react_public.txt` → `evaluate_react("public")`.
- Commit your results and share.

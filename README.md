# Unlikely Correlations

An open-source multi-agent research platform that surfaces hidden associations
between health outcomes and environmental, occupational, or geographic factors.
Powered by Claude.

## What it does

1. **Fetches data** from 20+ public US data sources at county level
2. **Finds associations** — Pearson/Spearman, partial correlation, PySAL LISA spatial clustering
3. **Searches the literature** — PubMed queries + Claude synthesis
4. **Assesses causation** — Bradford Hill criteria, confounders, alternatives
5. **Proposes a study** — investigation brief or full research proposal

## Pipeline

```
Data Ingestion -> Correlation Agent -> Literature Agent -> Causation Agent -> Study Design Agent
```

## Installation

```bash
git clone https://github.com/jt630/Huh-Really-.git
cd Huh-Really-
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Configuration

```bash
cp .env.example .env  # add your ANTHROPIC_API_KEY
```

## Run

```bash
streamlit run app.py
# or
python -m src.pipeline --case cases/parkinsons_golf.yaml
```

## Demo case

`cases/parkinsons_golf.yaml` — Parkinson's disease mortality vs. paraquat, rotenone,
chlorpyrifos, and golf course density at the US county level (2015-2019).

## License

MIT

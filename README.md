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

A synthetic fixture (`cases/parkinsons_golf_fixture.csv`) ships with the repo so
the demo runs fully offline except for the Claude API calls.

### 60-second dry run

1. `streamlit run app.py` — the sidebar loads with the Parkinson's hypothesis pre-filled.
2. Click **Demo** (right of the Run button). This loads the bundled case + fixture.
3. Watch the progress bar step through the four stages: `correlation → literature → causation → study_design`.
4. Step through the result tabs:
   - **Overview** — exposure, outcome, run status.
   - **Correlation** — Pearson/Spearman, partial correlation controlling for the
     selected confounders, p-values. Labeled "Source Data — Cited and Verifiable".
   - **Literature** — PubMed-derived supporting / contradicting counts,
     key findings, evidence gaps. Labeled "AI Analysis — Verify Independently".
   - **Causation** — Bradford Hill table, ecological-fallacy warning,
     confidence rating.
   - **Study Design** — full Markdown research brief, with a download button.
   - **Raw Data** — the full `PipelineResult` JSON for inspection.
5. Flip **Mode** to *Discovery Mode* to sweep the top-N exposures against the
   same outcome (uses the live data registry, so needs network + API keys).

### CLI equivalent

```bash
python -m src.pipeline --case cases/parkinsons_golf.yaml
```

Prints the same `PipelineResult` JSON to stdout; useful for screen-share demos
where Streamlit isn't convenient.

## License

MIT

import uuid, json, logging
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
from src.config import get_settings

class PipelineResult(BaseModel):
    run_id: str
    exposure: str
    outcome: str
    timestamp: str
    correlation: dict | None = None
    literature: dict | None = None
    causation: dict | None = None
    study_design: dict | None = None
    error: str | None = None
    demo_mode: bool = False

def _load_fixture(outputs_dir: Path | None, name: str) -> dict | None:
    """Return a pre-computed agent output JSON if present, else None."""
    if outputs_dir is None:
        return None
    p = outputs_dir / f"{name}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        logging.exception("Failed to parse fixture %s", p)
        return None

class Pipeline:
    def __init__(self):
        self.settings = get_settings()

    def run_hypothesis(self, exposure, outcome, confounders, output_mode="brief",
                       progress_cb=None, fixture_path: str | None = None,
                       fixture_outputs: str | None = None) -> PipelineResult:
        run_id = str(uuid.uuid4())[:8]
        outputs_dir = Path(fixture_outputs) if fixture_outputs else None
        demo_mode = outputs_dir is not None and outputs_dir.exists()
        result = PipelineResult(run_id=run_id, exposure=exposure, outcome=outcome,
                                timestamp=datetime.utcnow().isoformat(),
                                demo_mode=demo_mode)
        run_dir = self.settings.results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Anthropic client is only built when we actually need the live API.
        client = None

        try:
            # Stage 1: Correlation — always runs locally (no API calls).
            if progress_cb: progress_cb("correlation", "running")
            from src.agents.correlation import CorrelationAgent, CorrelationRequest
            import pandas as pd
            if fixture_path and Path(fixture_path).exists():
                df = pd.read_csv(fixture_path).set_index("county_fips")
                logging.info("Pipeline: loaded fixture from %s (%d rows)", fixture_path, len(df))
            else:
                df = pd.DataFrame()
                try:
                    from src.db import load_source, available as db_available
                    if db_available():
                        outcome_df = load_source(outcome.split("_")[0])
                        exposure_df = load_source(exposure.split("_")[0])
                        if outcome_df is not None and exposure_df is not None:
                            df = outcome_df.join(exposure_df, how="inner")
                            logging.info("Pipeline: loaded from DuckDB (%d rows)", len(df))
                except Exception:
                    pass
                if df.empty:
                    try:
                        from src.data.registry import DataSourceRegistry
                        reg = DataSourceRegistry()
                        outcome_df = reg.load(outcome.split("_")[0], variable=outcome)
                        exposure_df = reg.load(exposure.split("_")[0], variable=exposure)
                        df = outcome_df.join(exposure_df, how="inner")
                    except Exception:
                        pass

            agent = CorrelationAgent()
            if not df.empty:
                req = CorrelationRequest(outcome_col=outcome, exposure_col=exposure,
                                         confounder_cols=confounders, data=df)
                corr_out = agent.run(req)
                result.correlation = {k: str(v) for k, v in corr_out.items()
                                      if not hasattr(v, 'to_html')}
            if progress_cb: progress_cb("correlation", "done")

            def _get_client():
                nonlocal client
                if client is None:
                    import anthropic
                    if not self.settings.anthropic_api_key:
                        raise RuntimeError(
                            "ANTHROPIC_API_KEY is not set and no fixture output is available for this stage."
                        )
                    client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
                return client

            # Stage 2: Literature — fixture if present, else live API call.
            if progress_cb: progress_cb("literature", "running")
            lit_fixture = _load_fixture(outputs_dir, "literature")
            if lit_fixture is not None:
                result.literature = lit_fixture
                logging.info("Pipeline: literature loaded from fixture (demo mode)")
            else:
                from src.agents.literature import LiteratureAgent, LiteratureRequest
                hyp = f"Is there a causal relationship between {exposure} and {outcome}?"
                lit_req = LiteratureRequest(hypothesis=hyp,
                                            max_results=self.settings.literature_max_results)
                lit_agent = LiteratureAgent()
                lit_result = lit_agent.run(lit_req, _get_client())
                result.literature = lit_result.model_dump(mode="json")
            if progress_cb: progress_cb("literature", "done")

            # Stage 3: Causation — fixture if present, else live API call.
            if progress_cb: progress_cb("causation", "running")
            caus_fixture = _load_fixture(outputs_dir, "causation")
            if caus_fixture is not None:
                result.causation = caus_fixture
                logging.info("Pipeline: causation loaded from fixture (demo mode)")
            else:
                from src.agents.causation import CausationAgent, CausationRequest
                caus_req = CausationRequest(
                    correlation_result=result.correlation or {},
                    literature_result=result.literature or {},
                    exposure=exposure, outcome=outcome)
                caus_agent = CausationAgent()
                caus_result = caus_agent.run(caus_req, _get_client())
                result.causation = caus_result.model_dump(mode="json")
            if progress_cb: progress_cb("causation", "done")

            # Stage 4: Study Design — fixture if present, else live API call.
            if progress_cb: progress_cb("study_design", "running")
            sd_fixture = _load_fixture(outputs_dir, "study_design")
            if sd_fixture is not None:
                result.study_design = sd_fixture
                logging.info("Pipeline: study_design loaded from fixture (demo mode)")
            else:
                from src.agents.study_design import StudyDesignAgent, StudyDesignRequest
                sd_req = StudyDesignRequest(
                    causation_result=result.causation,
                    literature_result=result.literature,
                    exposure=exposure, outcome=outcome, output_mode=output_mode)
                sd_agent = StudyDesignAgent()
                sd_result = sd_agent.run(sd_req, _get_client())
                result.study_design = sd_result.model_dump(mode="json")
            if progress_cb: progress_cb("study_design", "done")

        except Exception as e:
            logging.exception("Pipeline error")
            result.error = str(e)

        (run_dir / "result.json").write_text(result.model_dump_json(indent=2))
        return result

    def run_discovery(self, outcome, top_n=20, confounders=None, output_mode="brief",
                      progress_cb=None) -> list[PipelineResult]:
        results = []
        try:
            from src.data.registry import DataSourceRegistry
            reg = DataSourceRegistry()
            all_vars = [col for cols in reg.list_variables().values() for col in cols]
            exposures = [v for v in all_vars if v != outcome][:top_n]
            for exp in exposures:
                r = self.run_hypothesis(exp, outcome, confounders or [], output_mode, progress_cb)
                results.append(r)
        except Exception:
            logging.exception("Discovery error")
        return results

def main():
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="Path to YAML case file")
    parser.add_argument("--exposure")
    parser.add_argument("--outcome")
    parser.add_argument("--confounders", nargs="*", default=[])
    parser.add_argument("--mode", default="brief")
    args = parser.parse_args()

    fixture_path = None
    fixture_outputs = None
    if args.case:
        import yaml
        case = yaml.safe_load(Path(args.case).read_text())
        exposure = case["exposures"][0]["name"]
        outcome = case["outcome"]
        confounders = [c["name"] for c in case.get("confounders", [])]
        fixture_path = case.get("fixture_data")
        fixture_outputs = case.get("fixture_outputs")
    else:
        exposure, outcome, confounders = args.exposure, args.outcome, args.confounders

    pipeline = Pipeline()
    result = pipeline.run_hypothesis(exposure, outcome, confounders, args.mode,
                                     progress_cb=lambda s, st: print(f"[{s}] {st}"),
                                     fixture_path=fixture_path,
                                     fixture_outputs=fixture_outputs)
    print(json.dumps(result.model_dump(mode="json"), indent=2))

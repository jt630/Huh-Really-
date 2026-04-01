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

class Pipeline:
    def __init__(self):
        self.settings = get_settings()

    def run_hypothesis(self, exposure, outcome, confounders, output_mode="brief",
                       progress_cb=None) -> PipelineResult:
        run_id = str(uuid.uuid4())[:8]
        result = PipelineResult(run_id=run_id, exposure=exposure, outcome=outcome,
                                timestamp=datetime.utcnow().isoformat())
        run_dir = self.settings.results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Stage 1: Correlation
            if progress_cb: progress_cb("correlation", "running")
            from src.agents.correlation import CorrelationAgent, CorrelationRequest
            import pandas as pd
            # Load data from registry
            try:
                from src.data.registry import DataSourceRegistry
                reg = DataSourceRegistry()
                outcome_df = reg.load(outcome.split("_")[0], variable=outcome)
                exposure_df = reg.load(exposure.split("_")[0], variable=exposure)
                df = outcome_df.join(exposure_df, how="inner")
            except Exception:
                df = pd.DataFrame()  # empty fallback for testing

            agent = CorrelationAgent()
            if not df.empty:
                req = CorrelationRequest(outcome_col=outcome, exposure_col=exposure,
                                         confounder_cols=confounders, data=df)
                corr_out = agent.run(req)
                result.correlation = {k: str(v) for k, v in corr_out.items()
                                      if not hasattr(v, 'to_html')}
            if progress_cb: progress_cb("correlation", "done")

            # Stage 2: Literature
            if progress_cb: progress_cb("literature", "running")
            import anthropic
            client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
            from src.agents.literature import LiteratureAgent, LiteratureRequest
            hyp = f"Is there a causal relationship between {exposure} and {outcome}?"
            lit_req = LiteratureRequest(hypothesis=hyp,
                                        max_results=self.settings.literature_max_results)
            lit_agent = LiteratureAgent()
            lit_result = lit_agent.run(lit_req, client)
            result.literature = lit_result.model_dump(mode="json")
            if progress_cb: progress_cb("literature", "done")

            # Stage 3: Causation
            if progress_cb: progress_cb("causation", "running")
            from src.agents.causation import CausationAgent, CausationRequest
            caus_req = CausationRequest(
                correlation_result=result.correlation or {},
                literature_result=result.literature or {},
                exposure=exposure, outcome=outcome)
            caus_agent = CausationAgent()
            caus_result = caus_agent.run(caus_req, client)
            result.causation = caus_result.model_dump(mode="json")
            if progress_cb: progress_cb("causation", "done")

            # Stage 4: Study Design
            if progress_cb: progress_cb("study_design", "running")
            from src.agents.study_design import StudyDesignAgent, StudyDesignRequest
            sd_req = StudyDesignRequest(
                causation_result=result.causation,
                literature_result=result.literature,
                exposure=exposure, outcome=outcome, output_mode=output_mode)
            sd_agent = StudyDesignAgent()
            sd_result = sd_agent.run(sd_req, client)
            result.study_design = sd_result.model_dump(mode="json")
            if progress_cb: progress_cb("study_design", "done")

        except Exception as e:
            logging.exception("Pipeline error")
            result.error = str(e)

        # Cache result
        (run_dir / "result.json").write_text(result.model_dump_json(indent=2))
        return result

    def run_discovery(self, outcome, top_n=20, confounders=None, output_mode="brief",
                      progress_cb=None) -> list[PipelineResult]:
        # Run sweep then run_hypothesis for top hits
        results = []
        try:
            from src.data.registry import DataSourceRegistry
            reg = DataSourceRegistry()
            exposures = [v for v in reg.list_variables() if v != outcome][:top_n]
            for exp in exposures:
                r = self.run_hypothesis(exp, outcome, confounders or [], output_mode, progress_cb)
                results.append(r)
        except Exception as e:
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

    if args.case:
        import yaml
        case = yaml.safe_load(Path(args.case).read_text())
        exposure = case["exposures"][0]["name"]
        outcome = case["outcome"]
        confounders = [c["name"] for c in case.get("confounders", [])]
    else:
        exposure, outcome, confounders = args.exposure, args.outcome, args.confounders

    pipeline = Pipeline()
    result = pipeline.run_hypothesis(exposure, outcome, confounders, args.mode,
                                     progress_cb=lambda s, st: print(f"[{s}] {st}"))
    print(json.dumps(result.model_dump(mode="json"), indent=2))

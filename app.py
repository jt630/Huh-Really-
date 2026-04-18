import os
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Unlikely Correlations", page_icon="🔬", layout="wide")

if not os.environ.get("ANTHROPIC_API_KEY"):
    from dotenv import load_dotenv
    load_dotenv()

has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

# Sidebar
with st.sidebar:
    st.title("🔬 Unlikely Correlations")
    mode = st.radio("Mode", ["Test Hypothesis", "Discovery Mode"])

    st.subheader("Variables")
    outcome = st.text_input("Outcome variable", value="parkinsons_mortality_rate")

    if mode == "Test Hypothesis":
        exposure = st.text_input("Exposure variable", value="paraquat_kg")
    else:
        top_n = st.slider("Top N exposures to sweep", 5, 50, 20)

    confounders = st.multiselect("Confounders",
        ["median_age", "median_income", "pct_rural", "healthcare_access"],
        default=["median_age", "median_income"])

    output_mode = st.selectbox("Output format", ["brief", "full"])

    col1, col2 = st.columns(2)
    run_btn = col1.button("▶ Run", type="primary")
    demo_btn = col2.button("Demo")

    st.divider()
    if has_api_key:
        st.caption("Live mode ready — ANTHROPIC_API_KEY detected.")
    else:
        st.caption("Demo mode only — no ANTHROPIC_API_KEY set. Click **Demo** to run the bundled case against pre-computed agent outputs.")

# Main area
st.title("Unlikely Correlations")
st.caption("Multi-agent research platform for detecting hidden health-environment associations.")

if demo_btn:
    import yaml
    case_path = Path("cases/parkinsons_golf.yaml")
    if case_path.exists():
        case = yaml.safe_load(case_path.read_text())
        st.session_state["demo_case"] = case
        st.session_state["demo_fixture"] = case.get("fixture_data")
        st.session_state["demo_outputs"] = case.get("fixture_outputs")
        st.session_state["run"] = True
        st.rerun()

should_run = run_btn or st.session_state.pop("run", False)
if should_run:
    if run_btn and not has_api_key:
        st.error(
            "**ANTHROPIC_API_KEY not set.** Live runs call the Claude API. "
            "Either create a `.env` with `ANTHROPIC_API_KEY=…` and restart, "
            "or click **Demo** to run the bundled case offline using pre-computed "
            "agent outputs."
        )
        st.stop()

    from src.pipeline import Pipeline

    progress = st.progress(0, text="Starting pipeline...")
    stages = ["correlation", "literature", "causation", "study_design"]
    stage_idx = {"correlation": 0, "literature": 1, "causation": 2, "study_design": 3}

    def update_progress(stage, status):
        idx = stage_idx.get(stage, 0)
        pct = int((idx + (1 if status == "done" else 0.5)) / len(stages) * 100)
        progress.progress(pct, text=f"{stage}: {status}")

    fixture_path = st.session_state.pop("demo_fixture", None)
    fixture_outputs = st.session_state.pop("demo_outputs", None)

    with st.spinner("Running pipeline..."):
        pipeline = Pipeline()
        try:
            if mode == "Test Hypothesis":
                result = pipeline.run_hypothesis(
                    exposure=exposure, outcome=outcome,
                    confounders=confounders, output_mode=output_mode,
                    progress_cb=update_progress,
                    fixture_path=fixture_path,
                    fixture_outputs=fixture_outputs)
                results = [result]
            else:
                results = pipeline.run_discovery(
                    outcome=outcome, top_n=top_n,
                    confounders=confounders, output_mode=output_mode,
                    progress_cb=update_progress)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            results = []

    progress.progress(100, text="Complete!")

    if results:
        result = results[0]

        if getattr(result, "demo_mode", False):
            st.success(
                "**Demo Mode** — agent outputs loaded from pre-computed fixtures "
                "(`cases/parkinsons_golf_outputs/`). Correlation stats are computed "
                "live from the CSV fixture; literature, causation, and study-design "
                "stages would call the Claude API in production. See README → "
                "*Demo Mode vs. Live Mode*."
            )

        tabs = st.tabs(["Overview", "Correlation", "Literature", "Causation", "Study Design", "Raw Data"])

        with tabs[0]:
            st.subheader("Summary")
            st.metric("Exposure", result.exposure)
            st.metric("Outcome", result.outcome)
            if result.error:
                st.error(result.error)

        with tabs[1]:
            st.subheader("Correlation Results")
            st.info("Source Data — Cited and Verifiable")
            if result.correlation:
                st.json(result.correlation)
            else:
                st.warning("No correlation data available.")

        with tabs[2]:
            st.subheader("Literature Synthesis")
            st.warning("AI Analysis — Verify Independently")
            if result.literature:
                synth = result.literature.get("synthesis", {})
                col1, col2, col3 = st.columns(3)
                col1.metric("Supporting Studies", synth.get("supporting_count", "—"))
                col2.metric("Contradicting Studies", synth.get("contradicting_count", "—"))
                col3.metric("Evidence Weight", synth.get("evidence_weight", "—"))
                if synth.get("key_findings"):
                    st.subheader("Key Findings")
                    for f in synth["key_findings"]:
                        st.write(f"• {f}")
                if synth.get("evidence_gaps"):
                    st.subheader("Evidence Gaps")
                    for g in synth["evidence_gaps"]:
                        st.write(f"• {g}")
            else:
                st.warning("No literature data available.")

        with tabs[3]:
            st.subheader("Causal Assessment")
            st.warning("AI Analysis — Verify Independently")
            if result.causation:
                st.error(result.causation.get("ecological_fallacy_warning", ""))
                st.metric("Overall Rating", result.causation.get("overall_rating", "—"))
                st.metric("Confidence", result.causation.get("confidence", "—"))
                bh = result.causation.get("bradford_hill", [])
                if bh:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(bh)[["name", "rating", "evidence"]])
            else:
                st.warning("No causation data available.")

        with tabs[4]:
            st.subheader("Study Design Proposal")
            st.warning("AI Analysis — Verify Independently")
            if result.study_design:
                md = result.study_design.get("markdown", "")
                if md:
                    st.markdown(md)
                    st.download_button("Download Markdown", md, file_name="proposal.md")
            else:
                st.warning("No study design available.")

        with tabs[5]:
            st.subheader("Raw Pipeline Output")
            st.info("Source Data — Cited and Verifiable")
            import json
            st.json(result.model_dump(mode="json") if hasattr(result, 'model_dump') else result.__dict__)

else:
    st.info("Configure parameters in the sidebar and click **Run** to start the pipeline, or click **Demo** to run the bundled Parkinson's / pesticide case against the offline fixture.")
    st.markdown(
        """
        **Pipeline stages**

        1. **Correlation** — Pearson/Spearman + partial correlation, optional PySAL LISA spatial clustering. *Runs locally, no API.*
        2. **Literature** — PubMed search + Claude synthesis of supporting / contradicting evidence. *API in live mode; fixture in demo mode.*
        3. **Causation** — Bradford Hill criteria, confounder review, alternative explanations. *API in live mode; fixture in demo mode.*
        4. **Study design** — Investigation brief or full research proposal (downloadable Markdown). *API in live mode; fixture in demo mode.*

        County-level analysis across the US (2015–2019 for the demo case).
        """
    )

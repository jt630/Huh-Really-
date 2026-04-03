import os
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Unlikely Correlations", page_icon="🔬", layout="wide")

if not os.environ.get("ANTHROPIC_API_KEY"):
    from dotenv import load_dotenv
    load_dotenv()
if not os.environ.get("ANTHROPIC_API_KEY"):
    st.error("**ANTHROPIC_API_KEY not set.** Create a `.env` file with `ANTHROPIC_API_KEY=your_key` then restart.")
    st.stop()

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
        st.session_state["run"] = True
        st.rerun()

if run_btn or st.session_state.get("run"):
    from src.pipeline import Pipeline

    progress = st.progress(0, text="Starting pipeline...")
    stages = ["correlation", "literature", "causation", "study_design"]
    stage_idx = {"correlation": 0, "literature": 1, "causation": 2, "study_design": 3}

    def update_progress(stage, status):
        idx = stage_idx.get(stage, 0)
        pct = int((idx + (1 if status == "done" else 0.5)) / len(stages) * 100)
        progress.progress(pct, text=f"{stage}: {status}")

    fixture_path = st.session_state.pop("demo_fixture", None)

    with st.spinner("Running pipeline..."):
        pipeline = Pipeline()
        try:
            if mode == "Test Hypothesis":
                result = pipeline.run_hypothesis(
                    exposure=exposure, outcome=outcome,
                    confounders=confounders, output_mode=output_mode,
                    progress_cb=update_progress, fixture_path=fixture_path)
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
        tabs = st.tabs(["Overview", "Correlation", "Literature", "Causation", "Study Design", "Raw Data"])

        result = results[0]  # primary result

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
    st.info("Configure parameters in the sidebar and click **Run** to start the pipeline.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/World_map_blank_without_borders.svg/1200px-World_map_blank_without_borders.svg.png",
             caption="County-level analysis across the US", use_column_width=True)

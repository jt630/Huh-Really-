"""
Tests for src/agents/study_design.py

All Claude API calls are mocked so the suite runs offline without any
ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agents.study_design import (
    DISCLAIMER,
    AlternativeDesign,
    GapResolution,
    StudyDesignAgent,
    StudyDesignRequest,
    StudyDesignResult,
    StudyProposal,
    generate_full_proposal,
    generate_investigation_brief,
    render_markdown,
)

BRIEF_PAYLOAD: dict = {
    "title": "Paraquat Exposure and Parkinson's Mortality: Rapid Investigation Brief",
    "study_type": "Ecological / Registry Linkage",
    "hypothesis": "Counties with higher agricultural paraquat use have elevated Parkinson's disease mortality rates.",
    "population": "US adult population (45+) residing in agricultural counties, 2010-2020.",
    "sample_size_rationale": "Ecological study uses all 3,143 US counties; no power calculation required.",
    "exposure_assessment": "USGS Pesticide National Synthesis Project annual kg-applied data by county.",
    "outcome_assessment": "CDC WONDER age-adjusted Parkinson's mortality (ICD-10 G20) by county.",
    "confounders": ["Median age", "Median household income", "Rural/urban classification"],
    "analysis_plan": "Immediate: flag high-exposure/high-mortality counties for follow-up.",
    "limitations": ["Ecological fallacy.", "Paraquat use data are modelled estimates."],
    "ethical_considerations": "No direct human subjects; secondary data only.",
    "timeline": "3 months for data linkage and preliminary analysis.",
    "resources": "1 epidemiologist, 1 data analyst; estimated $50,000.",
    "gap_resolutions": [{"uncertainty": "Individual-level exposure unknown", "how_addressed": "Registry linkage."}],
    "alternative_designs": [],
    "disclaimer": DISCLAIMER,
}

FULL_PAYLOAD: dict = {
    "title": "Prospective Cohort Investigation of Agricultural Paraquat Exposure and Parkinson's Disease",
    "study_type": "Prospective cohort.",
    "hypothesis": "Long-term residential proximity to high paraquat-use agricultural land increases Parkinson's disease hazard.",
    "population": "Adults aged 40-75 in top-quintile paraquat-use counties.",
    "sample_size_rationale": "80% power requires ~12,000 participants.",
    "exposure_assessment": "Annual paraquat kg-applied within 5 km residential radius.",
    "outcome_assessment": "Incident Parkinson's disease via linked EHR diagnostic codes (ICD-10 G20).",
    "confounders": ["Age (continuous)", "Sex", "Smoking history (pack-years)", "Median household income"],
    "analysis_plan": "Primary: Cox proportional-hazards model with time-varying paraquat exposure.",
    "limitations": ["Ecological-to-individual translation proxy.", "Loss to follow-up bias."],
    "ethical_considerations": "IRB approval required. Written informed consent for all participants.",
    "timeline": "Year 1: Protocol development. Years 2-3: Enrollment. Years 4-12: Follow-up.",
    "resources": "Estimated $8M over 13 years.",
    "gap_resolutions": [
        {"uncertainty": "Ecological fallacy", "how_addressed": "Individual-level exposure with biomarker validation."},
        {"uncertainty": "Temporality unclear", "how_addressed": "Prospective design with 10+ year follow-up."},
    ],
    "alternative_designs": [
        {"study_type": "Case-Control Study", "description": "Clinic-based case-control.", "strengths": ["Fast"], "weaknesses": ["Recall bias"]},
        {"study_type": "Mendelian Randomization", "description": "Use genetic variants as instrumental variables.", "strengths": ["Reduces confounding"], "weaknesses": ["Instruments unavailable"]},
    ],
    "disclaimer": DISCLAIMER,
}


def _make_mock_client(payload: dict) -> MagicMock:
    mock_content = MagicMock()
    mock_content.text = json.dumps(payload)
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def brief_request() -> StudyDesignRequest:
    return StudyDesignRequest(
        causation_result={"rating": "Possible", "confidence": "Moderate"},
        literature_result={"supporting": 12, "contradicting": 3},
        exposure="paraquat_kg",
        outcome="parkinsons_mortality_rate",
        output_mode="brief",
    )


@pytest.fixture
def full_request() -> StudyDesignRequest:
    return StudyDesignRequest(
        causation_result={"rating": "Probable", "confidence": "Moderate"},
        literature_result={"supporting": 20, "contradicting": 4},
        exposure="paraquat_kg",
        outcome="parkinsons_mortality_rate",
        output_mode="full",
    )


class TestPydanticModels:
    def test_gap_resolution(self):
        gr = GapResolution(uncertainty="Individual exposure unknown", how_addressed="Biomarker sub-study")
        assert gr.uncertainty == "Individual exposure unknown"

    def test_alternative_design(self):
        alt = AlternativeDesign(study_type="Case-Control", description="Clinic-based", strengths=["Fast"], weaknesses=["Recall bias"])
        assert len(alt.strengths) == 1

    def test_study_proposal_defaults(self):
        sp = StudyProposal(
            title="Test", study_type="Cohort", hypothesis="H", population="P",
            sample_size_rationale="S", exposure_assessment="E", outcome_assessment="O",
            analysis_plan="A", ethical_considerations="EC", timeline="T", resources="R",
        )
        assert sp.disclaimer == DISCLAIMER
        assert sp.confounders == []
        assert sp.gap_resolutions == []
        assert sp.alternative_designs == []

    def test_invalid_output_mode(self):
        with pytest.raises(Exception):
            StudyDesignRequest(causation_result={}, literature_result={}, exposure="x", outcome="y", output_mode="invalid")  # type: ignore[arg-type]


class TestGenerateInvestigationBrief:
    def test_returns_study_proposal(self, brief_request):
        result = generate_investigation_brief(brief_request, _make_mock_client(BRIEF_PAYLOAD))
        assert isinstance(result, StudyProposal)

    def test_fields_populated(self, brief_request):
        result = generate_investigation_brief(brief_request, _make_mock_client(BRIEF_PAYLOAD))
        assert result.title
        assert result.hypothesis
        assert result.disclaimer == DISCLAIMER

    def test_gap_resolutions_parsed(self, brief_request):
        result = generate_investigation_brief(brief_request, _make_mock_client(BRIEF_PAYLOAD))
        assert len(result.gap_resolutions) >= 1
        assert isinstance(result.gap_resolutions[0], GapResolution)

    def test_brief_has_no_alternative_designs(self, brief_request):
        result = generate_investigation_brief(brief_request, _make_mock_client(BRIEF_PAYLOAD))
        assert result.alternative_designs == []


class TestGenerateFullProposal:
    def test_returns_study_proposal(self, full_request):
        result = generate_full_proposal(full_request, _make_mock_client(FULL_PAYLOAD))
        assert isinstance(result, StudyProposal)

    def test_alternative_designs_populated(self, full_request):
        result = generate_full_proposal(full_request, _make_mock_client(FULL_PAYLOAD))
        assert len(result.alternative_designs) >= 2
        for alt in result.alternative_designs:
            assert isinstance(alt, AlternativeDesign)

    def test_gap_resolutions_populated(self, full_request):
        result = generate_full_proposal(full_request, _make_mock_client(FULL_PAYLOAD))
        assert len(result.gap_resolutions) >= 1

    def test_disclaimer_present(self, full_request):
        result = generate_full_proposal(full_request, _make_mock_client(FULL_PAYLOAD))
        assert result.disclaimer == DISCLAIMER


class TestRenderMarkdown:
    def test_brief_markdown_contains_title(self):
        md = render_markdown(StudyProposal(**BRIEF_PAYLOAD), "brief")
        assert BRIEF_PAYLOAD["title"] in md

    def test_brief_markdown_contains_disclaimer(self):
        md = render_markdown(StudyProposal(**BRIEF_PAYLOAD), "brief")
        assert DISCLAIMER in md

    def test_brief_markdown_contains_gap_table(self):
        md = render_markdown(StudyProposal(**BRIEF_PAYLOAD), "brief")
        assert "Gap Resolution" in md

    def test_full_markdown_has_alternative_designs_section(self):
        md = render_markdown(StudyProposal(**FULL_PAYLOAD), "full")
        assert "Alternative Study Designs" in md

    def test_full_markdown_contains_confounders(self):
        md = render_markdown(StudyProposal(**FULL_PAYLOAD), "full")
        assert "Age (continuous)" in md


class TestStudyDesignAgent:
    def test_run_brief_returns_result(self, brief_request):
        result = StudyDesignAgent(_make_mock_client(BRIEF_PAYLOAD)).run(brief_request)
        assert isinstance(result, StudyDesignResult)
        assert result.mode == "brief"

    def test_run_full_returns_result(self, full_request):
        result = StudyDesignAgent(_make_mock_client(FULL_PAYLOAD)).run(full_request)
        assert isinstance(result, StudyDesignResult)
        assert result.mode == "full"

    def test_markdown_rendered(self, brief_request):
        result = StudyDesignAgent(_make_mock_client(BRIEF_PAYLOAD)).run(brief_request)
        assert isinstance(result.markdown, str)
        assert len(result.markdown) > 50

    def test_proposal_disclaimer_in_markdown(self, brief_request):
        result = StudyDesignAgent(_make_mock_client(BRIEF_PAYLOAD)).run(brief_request)
        assert DISCLAIMER in result.markdown


class TestEdgeCases:
    def test_json_extraction_handles_markdown_fence(self):
        from src.agents.study_design import _extract_json
        payload = {"key": "value", "num": 42}
        result = _extract_json(f'```json\n{json.dumps(payload)}\n```')
        assert result == payload

    def test_json_extraction_raises_on_no_json(self):
        from src.agents.study_design import _extract_json
        with pytest.raises(ValueError):
            _extract_json("No JSON here at all.")

    def test_render_markdown_empty_lists(self):
        proposal = StudyProposal(
            title="Minimal", study_type="Cohort", hypothesis="H", population="P",
            sample_size_rationale="S", exposure_assessment="E", outcome_assessment="O",
            analysis_plan="A", ethical_considerations="EC", timeline="T", resources="R",
        )
        md = render_markdown(proposal, "full")
        assert "Minimal" in md
        assert DISCLAIMER in md

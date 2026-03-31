"""
Tests for src/agents/causation.py

Unit tests use mocked Anthropic responses; no real API calls are made.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agents.causation import (
    ECOLOGICAL_FALLACY_WARNING,
    AlternativeExplanation,
    BradfordHillCriterion,
    CausationAgent,
    CausationRequest,
    CausationResult,
    ConfounderAssessment,
    _BH_CRITERIA,
    _FIXED_ALTERNATIVES,
    _partial_correlation,
    analyze_confounders,
    assess_bradford_hill,
    evaluate_alternatives,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(response_text: str) -> MagicMock:
    """Return a mock Anthropic client that always returns *response_text*."""
    mock_content = MagicMock()
    mock_content.text = response_text
    mock_message = MagicMock()
    mock_message.content = [mock_content]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message
    return mock_client


_SAMPLE_BH_RESPONSE = json.dumps([
    {
        "name": name,
        "rating": "Moderate",
        "evidence": f"The data suggest a {name} relationship.",
        "gaps": f"Limited longitudinal data for {name}.",
    }
    for name in _BH_CRITERIA
])

_SAMPLE_ALT_RESPONSE = json.dumps([
    {
        "name": name,
        "description": f"Description for {name}.",
        "plausibility": "Moderate",
        "reasoning": f"The evidence is consistent with {name} being a concern.",
    }
    for name in _FIXED_ALTERNATIVES
])

_SAMPLE_SYNTH_RESPONSE = json.dumps({
    "overall_rating": "Possible",
    "confidence": "Moderate",
    "top_supporting": ["point A", "point B", "point C"],
    "top_uncertainties": ["gap X", "gap Y", "gap Z"],
    "recommended_next_steps": ["step 1", "step 2", "step 3"],
})

_SAMPLE_DOMAIN_CONF_RESPONSE = json.dumps(["age_distribution", "income_level"])


@pytest.fixture
def base_request() -> CausationRequest:
    return CausationRequest(
        correlation_result={"pearson_r": 0.62, "p_value": 0.001},
        literature_result={
            "supporting": ["Study A supports the association."],
            "contradicting": [],
            "gaps": ["Few longitudinal studies."],
        },
        exposure="paraquat_kg",
        outcome="parkinsons_mortality_rate",
        confounder_data=None,
    )


@pytest.fixture
def request_with_confounders(base_request) -> CausationRequest:
    rng = np.random.default_rng(0)
    n = 50
    paraquat = rng.uniform(0, 1000, n)
    age = rng.uniform(30, 80, n)
    income = rng.uniform(20_000, 100_000, n)
    # Make outcome weakly correlated with exposure and strongly with age
    parkinsons = 0.3 * paraquat / 1000 + 0.7 * age / 80 + rng.normal(0, 0.05, n)
    df = pd.DataFrame({
        "paraquat_kg": paraquat,
        "parkinsons_mortality_rate": parkinsons,
        "age_index": age,
        "income_index": income,
    })
    return base_request.model_copy(update={"confounder_data": df})


# ---------------------------------------------------------------------------
# Unit tests: _partial_correlation
# ---------------------------------------------------------------------------

class TestPartialCorrelation:
    def test_returns_float(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "x": rng.normal(size=30),
            "y": rng.normal(size=30),
            "z": rng.normal(size=30),
        })
        r = _partial_correlation(df, "x", "y", "z")
        assert isinstance(r, float)
        assert -1.0 <= r <= 1.0

    def test_too_few_rows_returns_nan(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "z": [1.0, 2.0]})
        r = _partial_correlation(df, "x", "y", "z")
        assert np.isnan(r)

    def test_controls_for_confounder(self):
        """When z causes both x and y, partial r should be near zero."""
        rng = np.random.default_rng(42)
        n = 200
        z = rng.normal(size=n)
        x = z + rng.normal(scale=0.1, size=n)
        y = z + rng.normal(scale=0.1, size=n)
        df = pd.DataFrame({"x": x, "y": y, "z": z})
        raw_r = float(np.corrcoef(x, y)[0, 1])
        partial_r = _partial_correlation(df, "x", "y", "z")
        # Partial correlation should be much closer to zero than raw
        assert abs(partial_r) < abs(raw_r)
        assert abs(partial_r) < 0.3


# ---------------------------------------------------------------------------
# Unit tests: assess_bradford_hill
# ---------------------------------------------------------------------------

class TestAssessBradfordHill:
    def test_returns_nine_criteria(self, base_request):
        client = _make_client(_SAMPLE_BH_RESPONSE)
        result = assess_bradford_hill(base_request, client)
        assert len(result) == 9
        assert all(isinstance(c, BradfordHillCriterion) for c in result)

    def test_criterion_names_match(self, base_request):
        client = _make_client(_SAMPLE_BH_RESPONSE)
        result = assess_bradford_hill(base_request, client)
        names = [c.name for c in result]
        for expected in _BH_CRITERIA:
            assert expected in names

    def test_ratings_are_valid(self, base_request):
        client = _make_client(_SAMPLE_BH_RESPONSE)
        result = assess_bradford_hill(base_request, client)
        valid = {"Strong", "Moderate", "Weak", "Insufficient"}
        for c in result:
            assert c.rating in valid

    def test_strips_markdown_fences(self, base_request):
        wrapped = f"```json\n{_SAMPLE_BH_RESPONSE}\n```"
        client = _make_client(wrapped)
        result = assess_bradford_hill(base_request, client)
        assert len(result) == 9

    def test_evidence_does_not_contain_prove(self, base_request):
        client = _make_client(_SAMPLE_BH_RESPONSE)
        result = assess_bradford_hill(base_request, client)
        for c in result:
            assert "prove" not in c.evidence.lower()

    def test_client_called_once(self, base_request):
        client = _make_client(_SAMPLE_BH_RESPONSE)
        assess_bradford_hill(base_request, client)
        assert client.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# Unit tests: analyze_confounders
# ---------------------------------------------------------------------------

class TestAnalyzeConfounders:
    def test_no_confounder_data_returns_empty_without_client(self, base_request):
        result = analyze_confounders(base_request, client=None)
        assert result == []

    def test_returns_confounder_assessment_objects(self, request_with_confounders):
        result = analyze_confounders(request_with_confounders, client=None)
        assert len(result) == 2  # age_index, income_index
        assert all(isinstance(a, ConfounderAssessment) for a in result)

    def test_pct_change_is_non_negative(self, request_with_confounders):
        result = analyze_confounders(request_with_confounders, client=None)
        for a in result:
            assert a.pct_change >= 0.0

    def test_survives_adjustment_is_bool(self, request_with_confounders):
        result = analyze_confounders(request_with_confounders, client=None)
        for a in result:
            assert isinstance(a.survives_adjustment, bool)

    def test_domain_confounders_added_with_client(self, base_request):
        client = _make_client(_SAMPLE_DOMAIN_CONF_RESPONSE)
        result = analyze_confounders(base_request, client=client)
        names = [a.confounder for a in result]
        assert "age_distribution" in names
        assert "income_level" in names

    def test_strong_confounder_reduces_r(self, request_with_confounders):
        """age_index confounds the relationship; adjusted_r should differ."""
        result = analyze_confounders(request_with_confounders, client=None)
        age_entry = next(a for a in result if a.confounder == "age_index")
        assert age_entry.pct_change > 0


# ---------------------------------------------------------------------------
# Unit tests: evaluate_alternatives
# ---------------------------------------------------------------------------

class TestEvaluateAlternatives:
    def test_returns_five_or_more(self, base_request):
        client = _make_client(_SAMPLE_ALT_RESPONSE)
        result = evaluate_alternatives(base_request, client)
        assert len(result) >= 5

    def test_all_fixed_alternatives_present(self, base_request):
        client = _make_client(_SAMPLE_ALT_RESPONSE)
        result = evaluate_alternatives(base_request, client)
        names = [a.name for a in result]
        for req in _FIXED_ALTERNATIVES:
            assert req in names

    def test_plausibility_values_valid(self, base_request):
        client = _make_client(_SAMPLE_ALT_RESPONSE)
        result = evaluate_alternatives(base_request, client)
        valid = {"High", "Moderate", "Low"}
        for a in result:
            assert a.plausibility in valid

    def test_missing_alternative_filled_in(self, base_request):
        """If model omits an alternative, it should be added with Insufficient."""
        partial = json.dumps([
            {
                "name": name,
                "description": "desc",
                "plausibility": "Low",
                "reasoning": "reasoning",
            }
            for name in _FIXED_ALTERNATIVES[:3]  # Omit last two
        ])
        client = _make_client(partial)
        result = evaluate_alternatives(base_request, client)
        names = [a.name for a in result]
        for req in _FIXED_ALTERNATIVES:
            assert req in names


# ---------------------------------------------------------------------------
# Integration test: CausationAgent.run
# ---------------------------------------------------------------------------

class TestCausationAgent:
    def _make_multi_response_client(
        self,
        bh_resp: str,
        domain_conf_resp: str,
        alt_resp: str,
        synth_resp: str,
    ) -> MagicMock:
        """Return a client whose create() cycles through provided responses."""
        responses = [bh_resp, domain_conf_resp, alt_resp, synth_resp]
        call_count = [0]

        def side_effect(**kwargs):
            idx = call_count[0] % len(responses)
            call_count[0] += 1
            mock_content = MagicMock()
            mock_content.text = responses[idx]
            mock_message = MagicMock()
            mock_message.content = [mock_content]
            return mock_message

        client = MagicMock()
        client.messages.create.side_effect = side_effect
        return client

    def test_run_returns_causation_result(self, base_request):
        client = self._make_multi_response_client(
            _SAMPLE_BH_RESPONSE,
            _SAMPLE_DOMAIN_CONF_RESPONSE,
            _SAMPLE_ALT_RESPONSE,
            _SAMPLE_SYNTH_RESPONSE,
        )
        agent = CausationAgent(client)
        result = agent.run(base_request)
        assert isinstance(result, CausationResult)

    def test_ecological_fallacy_warning_always_set(self, base_request):
        client = self._make_multi_response_client(
            _SAMPLE_BH_RESPONSE,
            _SAMPLE_DOMAIN_CONF_RESPONSE,
            _SAMPLE_ALT_RESPONSE,
            _SAMPLE_SYNTH_RESPONSE,
        )
        agent = CausationAgent(client)
        result = agent.run(base_request)
        assert result.ecological_fallacy_warning == ECOLOGICAL_FALLACY_WARNING
        assert len(result.ecological_fallacy_warning) > 0

    def test_ecological_warning_mentions_county(self, base_request):
        client = self._make_multi_response_client(
            _SAMPLE_BH_RESPONSE,
            _SAMPLE_DOMAIN_CONF_RESPONSE,
            _SAMPLE_ALT_RESPONSE,
            _SAMPLE_SYNTH_RESPONSE,
        )
        agent = CausationAgent(client)
        result = agent.run(base_request)
        assert "county" in result.ecological_fallacy_warning.lower()

    def test_overall_rating_valid(self, base_request):
        client = self._make_multi_response_client(
            _SAMPLE_BH_RESPONSE,
            _SAMPLE_DOMAIN_CONF_RESPONSE,
            _SAMPLE_ALT_RESPONSE,
            _SAMPLE_SYNTH_RESPONSE,
        )
        agent = CausationAgent(client)
        result = agent.run(base_request)
        valid = {"Probable", "Possible", "Insufficient", "Likely non-causal"}
        assert result.overall_rating in valid

    def test_confidence_valid(self, base_request):
        client = self._make_multi_response_client(
            _SAMPLE_BH_RESPONSE,
            _SAMPLE_DOMAIN_CONF_RESPONSE,
            _SAMPLE_ALT_RESPONSE,
            _SAMPLE_SYNTH_RESPONSE,
        )
        agent = CausationAgent(client)
        result = agent.run(base_request)
        assert result.confidence in {"High", "Moderate", "Low"}

    def test_bradford_hill_nine_criteria(self, base_request):
        client = self._make_multi_response_client(
            _SAMPLE_BH_RESPONSE,
            _SAMPLE_DOMAIN_CONF_RESPONSE,
            _SAMPLE_ALT_RESPONSE,
            _SAMPLE_SYNTH_RESPONSE,
        )
        agent = CausationAgent(client)
        result = agent.run(base_request)
        assert len(result.bradford_hill) == 9

    def test_top_supporting_is_list_of_strings(self, base_request):
        client = self._make_multi_response_client(
            _SAMPLE_BH_RESPONSE,
            _SAMPLE_DOMAIN_CONF_RESPONSE,
            _SAMPLE_ALT_RESPONSE,
            _SAMPLE_SYNTH_RESPONSE,
        )
        agent = CausationAgent(client)
        result = agent.run(base_request)
        assert isinstance(result.top_supporting, list)
        assert all(isinstance(s, str) for s in result.top_supporting)

    def test_run_with_confounder_data(self, request_with_confounders):
        client = self._make_multi_response_client(
            _SAMPLE_BH_RESPONSE,
            _SAMPLE_DOMAIN_CONF_RESPONSE,
            _SAMPLE_ALT_RESPONSE,
            _SAMPLE_SYNTH_RESPONSE,
        )
        agent = CausationAgent(client)
        result = agent.run(request_with_confounders)
        # Should have statistical confounders + domain confounders
        assert len(result.confounders) >= 2

    def test_recommended_next_steps_non_empty(self, base_request):
        client = self._make_multi_response_client(
            _SAMPLE_BH_RESPONSE,
            _SAMPLE_DOMAIN_CONF_RESPONSE,
            _SAMPLE_ALT_RESPONSE,
            _SAMPLE_SYNTH_RESPONSE,
        )
        agent = CausationAgent(client)
        result = agent.run(base_request)
        assert len(result.recommended_next_steps) >= 1

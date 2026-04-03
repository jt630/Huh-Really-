"""
Causation Agent - Prompt 5.

Capabilities:
- Bradford Hill criteria assessment (all 9)
- Partial-correlation confounder analysis
- Alternative explanation evaluation
- Overall causal rating with confidence
- Ecological fallacy warning on every assessment

[CO-DESIGN] Requires domain expert review before production use.
"""
from __future__ import annotations

import json
import textwrap
from typing import TYPE_CHECKING, Literal

import anthropic
import numpy as np
import pandas as pd
from pydantic import BaseModel

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class BradfordHillCriterion(BaseModel):
    name: str
    rating: Literal["Strong", "Moderate", "Weak", "Insufficient"]
    evidence: str
    gaps: str


class ConfounderAssessment(BaseModel):
    confounder: str
    unadjusted_r: float
    adjusted_r: float
    pct_change: float
    survives_adjustment: bool


class AlternativeExplanation(BaseModel):
    name: str
    description: str
    plausibility: Literal["High", "Moderate", "Low"]
    reasoning: str


class CausationRequest(BaseModel):
    """Input bundle for all causation-assessment steps."""

    model_config = {"arbitrary_types_allowed": True}

    # Outputs from previous agents (passed as dicts / plain objects)
    correlation_result: dict  # serialisable summary from CorrelationAgent
    literature_result: dict   # serialisable summary from LiteratureAgent

    exposure: str
    outcome: str

    # Optional DataFrame with confounder columns (rows = observations)
    confounder_data: pd.DataFrame | None = None


class CausationResult(BaseModel):
    bradford_hill: list[BradfordHillCriterion]
    confounders: list[ConfounderAssessment]
    alternatives: list[AlternativeExplanation]
    overall_rating: Literal["Probable", "Possible", "Insufficient", "Likely non-causal"]
    confidence: Literal["High", "Moderate", "Low"]
    top_supporting: list[str]
    top_uncertainties: list[str]
    ecological_fallacy_warning: str
    recommended_next_steps: list[str]


# ---------------------------------------------------------------------------
# Bradford Hill assessment
# ---------------------------------------------------------------------------

_BH_SYSTEM = textwrap.dedent("""\
    You are an epidemiologist evaluating a potential causal relationship.
    You MUST NOT use the word "prove" or any form of it.
    Instead use language such as "support", "consistent with", "suggest",
    "indicate", "is compatible with".
    Respond ONLY with valid JSON — no markdown fences, no prose outside JSON.
""")

_BH_CRITERIA = [
    "strength",
    "consistency",
    "specificity",
    "temporality",
    "biological_gradient",
    "plausibility",
    "coherence",
    "experiment",
    "analogy",
]


def assess_bradford_hill(
    request: CausationRequest,
    client: anthropic.Anthropic,
) -> list[BradfordHillCriterion]:
    """Ask Claude to rate all 9 Bradford-Hill criteria.

    Parameters
    ----------
    request:
        The CausationRequest with correlation stats and literature synthesis.
    client:
        An initialised Anthropic client.

    Returns
    -------
    list[BradfordHillCriterion]
        One entry per criterion, ordered as ``_BH_CRITERIA``.
    """
    user_prompt = textwrap.dedent(f"""\
        Exposure  : {request.exposure}
        Outcome   : {request.outcome}

        === Correlation statistics ===
        {json.dumps(request.correlation_result, indent=2)}

        === Literature synthesis ===
        {json.dumps(request.literature_result, indent=2)}

        Evaluate EACH of the following Bradford-Hill criteria for this
        exposure-outcome pair.  For every criterion provide:
          - rating  : one of "Strong", "Moderate", "Weak", "Insufficient"
          - evidence : specific evidence that supports the rating (use
                       language such as "support", "consistent with",
                       "suggest" — never "prove")
          - gaps    : key evidence gaps that limit the rating

        Criteria to evaluate (use exactly these names as the "name" field):
        {json.dumps(_BH_CRITERIA)}

        Return a JSON array of objects, one per criterion:
        [
          {{"name": "strength", "rating": "...", "evidence": "...", "gaps": "..."}},
          ...
        ]
    """)

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        system=_BH_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = message.content[0].text.strip()
    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    data = json.loads(raw)
    return [BradfordHillCriterion(**item) for item in data]


# ---------------------------------------------------------------------------
# Confounder analysis
# ---------------------------------------------------------------------------

def _partial_correlation(
    df: pd.DataFrame,
    exposure_col: str,
    outcome_col: str,
    confounder_col: str,
) -> float:
    """Compute Pearson partial correlation between exposure and outcome,
    controlling for a single confounder, via residual regression."""
    subset = df[[exposure_col, outcome_col, confounder_col]].dropna()
    if len(subset) < 4:
        return float("nan")

    x = subset[exposure_col].to_numpy(dtype=float)
    y = subset[outcome_col].to_numpy(dtype=float)
    z = subset[confounder_col].to_numpy(dtype=float)

    # Residualise x on z
    z_dm = z - z.mean()
    beta_xz = np.dot(z_dm, x) / np.dot(z_dm, z_dm) if np.dot(z_dm, z_dm) != 0 else 0.0
    res_x = x - (z_dm * beta_xz + x.mean())

    # Residualise y on z
    beta_yz = np.dot(z_dm, y) / np.dot(z_dm, z_dm) if np.dot(z_dm, z_dm) != 0 else 0.0
    res_y = y - (z_dm * beta_yz + y.mean())

    denom = np.std(res_x) * np.std(res_y)
    if denom == 0:
        return float("nan")
    return float(np.corrcoef(res_x, res_y)[0, 1])


def _identify_domain_confounders(
    request: CausationRequest,
    client: anthropic.Anthropic,
) -> list[str]:
    """Ask Claude to suggest domain-specific confounders not in the dataset."""
    system = (
        "You are an epidemiologist.  List plausible confounders for the "
        "given exposure-outcome relationship that are NOT in the provided "
        "dataset.  Respond ONLY with a JSON array of short confounder names."
    )
    user = textwrap.dedent(f"""\
        Exposure : {request.exposure}
        Outcome  : {request.outcome}
        Confounders already in dataset: {list(request.confounder_data.columns) if request.confounder_data is not None else []}

        List up to 8 additional domain-specific confounders not in the dataset.
        Return only a JSON array of strings, e.g. ["age", "sex", ...].
    """)
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def analyze_confounders(
    request: CausationRequest,
    client: anthropic.Anthropic | None = None,
) -> list[ConfounderAssessment]:
    """Compute partial-correlation adjustments for each confounder column.

    If *client* is provided, also asks Claude for domain-specific confounders
    that are not present in the dataset (added as assessments with NaN r values
    and ``survives_adjustment=True`` to flag they need investigation).

    Parameters
    ----------
    request:
        CausationRequest; ``confounder_data`` must contain the exposure column
        (``request.exposure``) and outcome column (``request.outcome``) plus
        any confounder columns.
    client:
        Optional Anthropic client for domain-confounder identification.

    Returns
    -------
    list[ConfounderAssessment]
    """
    results: list[ConfounderAssessment] = []

    if request.confounder_data is not None and not request.confounder_data.empty:
        df = request.confounder_data
        exposure_col = request.exposure
        outcome_col = request.outcome

        # Confirm exposure and outcome are present
        if exposure_col in df.columns and outcome_col in df.columns:
            # Unadjusted correlation
            subset = df[[exposure_col, outcome_col]].dropna()
            if len(subset) >= 2:
                unadjusted_r = float(np.corrcoef(
                    subset[exposure_col].to_numpy(dtype=float),
                    subset[outcome_col].to_numpy(dtype=float),
                )[0, 1])
            else:
                unadjusted_r = float("nan")

            # Per-confounder partial correlation
            confounder_cols = [
                c for c in df.columns if c not in (exposure_col, outcome_col)
            ]
            for conf in confounder_cols:
                adj_r = _partial_correlation(df, exposure_col, outcome_col, conf)
                if not np.isnan(unadjusted_r) and not np.isnan(adj_r) and unadjusted_r != 0:
                    pct = abs((adj_r - unadjusted_r) / unadjusted_r) * 100
                else:
                    pct = 0.0
                survives = abs(adj_r) >= 0.1 if not np.isnan(adj_r) else True
                results.append(ConfounderAssessment(
                    confounder=conf,
                    unadjusted_r=round(unadjusted_r, 4),
                    adjusted_r=round(adj_r, 4) if not np.isnan(adj_r) else 0.0,
                    pct_change=round(pct, 2),
                    survives_adjustment=survives,
                ))

    # Domain-specific confounders from Claude
    if client is not None:
        domain_confounders = _identify_domain_confounders(request, client)
        for dc in domain_confounders:
            results.append(ConfounderAssessment(
                confounder=dc,
                unadjusted_r=0.0,
                adjusted_r=0.0,
                pct_change=0.0,
                survives_adjustment=True,  # Unknown — needs investigation
            ))

    return results


# ---------------------------------------------------------------------------
# Alternative explanations
# ---------------------------------------------------------------------------

_ALT_SYSTEM = textwrap.dedent("""\
    You are an epidemiologist evaluating alternative explanations for an
    observed association.  You MUST NOT use the word "prove".
    Respond ONLY with valid JSON — no markdown fences, no prose outside JSON.
""")

_FIXED_ALTERNATIVES = [
    "ecological_fallacy",
    "selection_bias",
    "information_bias",
    "reverse_causation",
    "shared_upstream_cause",
]


def evaluate_alternatives(
    request: CausationRequest,
    client: anthropic.Anthropic,
) -> list[AlternativeExplanation]:
    """Evaluate plausibility of standard alternative explanations.

    Always evaluates the five canonical alternatives.

    Parameters
    ----------
    request:
        The CausationRequest.
    client:
        An initialised Anthropic client.

    Returns
    -------
    list[AlternativeExplanation]
    """
    user_prompt = textwrap.dedent(f"""\
        Exposure : {request.exposure}
        Outcome  : {request.outcome}

        === Correlation statistics ===
        {json.dumps(request.correlation_result, indent=2)}

        === Literature synthesis ===
        {json.dumps(request.literature_result, indent=2)}

        Evaluate the plausibility of EACH of the following alternative
        explanations for the observed association:

        1. ecological_fallacy   – The association at county/population level
                                  does not hold at the individual level.
        2. selection_bias       – Systematic differences in who was included
                                  in the study population.
        3. information_bias     – Measurement error in exposure or outcome.
        4. reverse_causation    – The outcome (or a precursor) influences
                                  the exposure rather than vice-versa.
        5. shared_upstream_cause – A third variable independently causes
                                  both the exposure and outcome.

        For each, provide:
          - name        : use exactly the snake_case names above
          - description : one sentence describing this explanation in context
          - plausibility: one of "High", "Moderate", "Low"
          - reasoning   : 2-3 sentences supporting the rating
                          (use "support", "suggest", "consistent with"
                           — never "prove")

        Return a JSON array of 5 objects:
        [
          {{"name": "ecological_fallacy", "description": "...",
            "plausibility": "...", "reasoning": "..."}},
          ...
        ]
    """)

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        system=_ALT_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    data = json.loads(raw)
    # Ensure all fixed alternatives are present
    returned_names = {item["name"] for item in data}
    for required in _FIXED_ALTERNATIVES:
        if required not in returned_names:
            data.append({
                "name": required,
                "description": "Not assessed by model.",
                "plausibility": "Insufficient",
                "reasoning": "Model did not return an assessment for this criterion.",
            })
    return [AlternativeExplanation(**item) for item in data]


# ---------------------------------------------------------------------------
# Overall synthesis
# ---------------------------------------------------------------------------

_SYNTH_SYSTEM = textwrap.dedent("""\
    You are an epidemiologist producing a final causal synthesis.
    You MUST NOT use the word "prove".
    Respond ONLY with valid JSON — no markdown fences, no prose outside JSON.
""")


def _synthesise(
    request: CausationRequest,
    bh_criteria: list[BradfordHillCriterion],
    confounders: list[ConfounderAssessment],
    alternatives: list[AlternativeExplanation],
    client: anthropic.Anthropic,
) -> dict:
    """Ask Claude for an overall causal rating and supporting details."""
    bh_summary = [{"name": c.name, "rating": c.rating} for c in bh_criteria]
    conf_summary = [
        {"confounder": c.confounder, "survives": c.survives_adjustment}
        for c in confounders
    ]
    alt_summary = [{"name": a.name, "plausibility": a.plausibility} for a in alternatives]

    user_prompt = textwrap.dedent(f"""\
        Exposure : {request.exposure}
        Outcome  : {request.outcome}

        Bradford-Hill ratings:
        {json.dumps(bh_summary, indent=2)}

        Confounder adjustments:
        {json.dumps(conf_summary, indent=2)}

        Alternative explanations:
        {json.dumps(alt_summary, indent=2)}

        Based on ALL of the above, provide:
          - overall_rating  : one of "Probable", "Possible",
                              "Insufficient", "Likely non-causal"
          - confidence      : one of "High", "Moderate", "Low"
          - top_supporting  : list of 3 strongest supporting points
                              (strings)
          - top_uncertainties: list of 3 biggest uncertainties (strings)
          - recommended_next_steps: list of 3-5 concrete next research
                                    steps (strings)

        Return a single JSON object:
        {{
          "overall_rating": "...",
          "confidence": "...",
          "top_supporting": ["...", "...", "..."],
          "top_uncertainties": ["...", "...", "..."],
          "recommended_next_steps": ["...", ...]
        }}
    """)

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=_SYNTH_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

ECOLOGICAL_FALLACY_WARNING = (
    "This is an ecological study. Associations at the county level cannot be "
    "assumed to hold at the individual level."
)


class CausationAgent:
    """Orchestrates Bradford-Hill assessment, confounder analysis,
    alternative explanation evaluation, and overall causal synthesis.
    """

    def run(self, request: CausationRequest, client: anthropic.Anthropic) -> CausationResult:
        """Run the full causation-assessment pipeline.

        Parameters
        ----------
        request:
            A fully populated ``CausationRequest``.

        Returns
        -------
        CausationResult
            Complete causal assessment; ``ecological_fallacy_warning`` is
            always non-empty.
        """
        # Step 1 – Bradford-Hill criteria
        bh_criteria = assess_bradford_hill(request, client)

        # Step 2 – Confounder analysis (statistical + domain-specific)
        confounders = analyze_confounders(request, client)

        # Step 3 – Alternative explanations
        alternatives = evaluate_alternatives(request, client)

        # Step 4 – Overall synthesis
        synthesis = _synthesise(
            request, bh_criteria, confounders, alternatives, client
        )

        return CausationResult(
            bradford_hill=bh_criteria,
            confounders=confounders,
            alternatives=alternatives,
            overall_rating=synthesis["overall_rating"],
            confidence=synthesis["confidence"],
            top_supporting=synthesis.get("top_supporting", []),
            top_uncertainties=synthesis.get("top_uncertainties", []),
            ecological_fallacy_warning=ECOLOGICAL_FALLACY_WARNING,
            recommended_next_steps=synthesis.get("recommended_next_steps", []),
        )

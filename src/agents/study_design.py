"""
Study Design Agent - Prompt 6.

Capabilities:
- Mode 1: Rapid Investigation Brief (1-2 pages)
- Mode 2: Full Research Proposal (5-10 pages)
- Gap resolution table
- Alternative study designs with trade-offs
- Export to Markdown

[CO-DESIGN] Requires domain expert review before production use.
"""
from __future__ import annotations

import json
import textwrap
from typing import Literal

import anthropic
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class GapResolution(BaseModel):
    uncertainty: str
    how_addressed: str


class AlternativeDesign(BaseModel):
    study_type: str
    description: str
    strengths: list[str]
    weaknesses: list[str]


class StudyProposal(BaseModel):
    title: str
    study_type: str
    hypothesis: str
    population: str
    sample_size_rationale: str
    exposure_assessment: str
    outcome_assessment: str
    confounders: list[str]
    analysis_plan: str
    limitations: list[str]
    ethical_considerations: str
    timeline: str
    resources: str
    gap_resolutions: list[GapResolution]
    alternative_designs: list[AlternativeDesign]
    disclaimer: str


class StudyDesignRequest(BaseModel):
    causation_result: dict
    literature_result: dict
    exposure: str
    outcome: str
    output_mode: Literal["brief", "full"] = "brief"


class StudyDesignResult(BaseModel):
    request: StudyDesignRequest
    proposal: StudyProposal
    markdown: str
    mode: str


# ---------------------------------------------------------------------------
# Brief generation
# ---------------------------------------------------------------------------

_BRIEF_SYSTEM = (
    "You are a public health researcher writing a rapid investigation brief. "
    "Return ONLY valid JSON."
)


def generate_investigation_brief(
    request: StudyDesignRequest,
    client: anthropic.Anthropic,
) -> StudyProposal:
    """Generate a rapid investigation brief using Claude.

    Parameters
    ----------
    request:
        The StudyDesignRequest containing causation and literature results.
    client:
        An initialised Anthropic client.

    Returns
    -------
    StudyProposal
        A StudyProposal populated from the model response with
        study_type="Rapid Investigation".
    """
    user_prompt = textwrap.dedent(f"""\
        Exposure : {request.exposure}
        Outcome  : {request.outcome}

        === Causation assessment summary ===
        {json.dumps(request.causation_result, indent=2)}

        === Literature synthesis summary ===
        {json.dumps(request.literature_result, indent=2)}

        Write a rapid investigation brief for this exposure-outcome relationship.
        Return a single JSON object with EXACTLY these fields:

        {{
          "title": "Brief descriptive title",
          "finding_summary": "2-3 sentence summary of the main finding",
          "evidence_assessment": "2-3 sentence assessment of the current evidence quality",
          "affected_populations": "Description of who may be affected",
          "recommended_actions": {{
            "immediate": ["action 1", "action 2"],
            "short_term": ["action 1", "action 2"],
            "long_term": ["action 1", "action 2"]
          }},
          "limitations": ["limitation 1", "limitation 2", "limitation 3"]
        }}

        Use language such as "suggest", "indicate", "consistent with" — never "prove".
    """)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=_BRIEF_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rstrip("`").strip()

    data = json.loads(raw)

    recommended_actions = data.get("recommended_actions", {})
    if isinstance(recommended_actions, dict):
        all_actions = (
            recommended_actions.get("immediate", [])
            + recommended_actions.get("short_term", [])
            + recommended_actions.get("long_term", [])
        )
        analysis_plan_text = (
            "Immediate actions: "
            + "; ".join(recommended_actions.get("immediate", []))
            + ". Short-term: "
            + "; ".join(recommended_actions.get("short_term", []))
            + ". Long-term: "
            + "; ".join(recommended_actions.get("long_term", []))
        )
    else:
        all_actions = recommended_actions if isinstance(recommended_actions, list) else []
        analysis_plan_text = "; ".join(all_actions)

    return StudyProposal(
        title=data.get("title", f"Rapid Investigation: {request.exposure} and {request.outcome}"),
        study_type="Rapid Investigation",
        hypothesis=data.get("finding_summary", ""),
        population=data.get("affected_populations", ""),
        sample_size_rationale="Not applicable for a rapid investigation brief.",
        exposure_assessment=f"Exposure: {request.exposure}",
        outcome_assessment=f"Outcome: {request.outcome}",
        confounders=[],
        analysis_plan=analysis_plan_text,
        limitations=data.get("limitations", []),
        ethical_considerations="Expert review required before informing policy decisions.",
        timeline="Rapid (days to weeks)",
        resources="Existing data sources and literature review.",
        gap_resolutions=[],
        alternative_designs=[],
        disclaimer=(
            "Generated proposal \u2014 requires expert review and independent "
            "verification before informing policy decisions."
        ),
    )


# ---------------------------------------------------------------------------
# Full proposal generation
# ---------------------------------------------------------------------------

_FULL_SYSTEM = (
    "You are an academic epidemiologist writing a full research proposal. "
    "Return ONLY valid JSON."
)


def generate_full_proposal(
    request: StudyDesignRequest,
    client: anthropic.Anthropic,
) -> StudyProposal:
    """Generate a full academic research proposal using Claude.

    Parameters
    ----------
    request:
        The StudyDesignRequest containing causation and literature results.
    client:
        An initialised Anthropic client.

    Returns
    -------
    StudyProposal
        A fully populated StudyProposal including gap resolutions and
        alternative designs.
    """
    top_uncertainties = request.causation_result.get("top_uncertainties", [])

    user_prompt = textwrap.dedent(f"""\
        Exposure : {request.exposure}
        Outcome  : {request.outcome}

        === Causation assessment summary ===
        {json.dumps(request.causation_result, indent=2)}

        === Literature synthesis summary ===
        {json.dumps(request.literature_result, indent=2)}

        Write a full academic research proposal for this exposure-outcome relationship.
        Return a single JSON object with EXACTLY these fields:

        {{
          "title": "Full descriptive academic title",
          "study_type": "e.g. Prospective Cohort Study / Case-Control / RCT / Cross-sectional",
          "hypothesis": "Formal null and alternative hypothesis statement",
          "population": "Detailed target population description including inclusion/exclusion criteria",
          "sample_size_rationale": "Statistical justification for sample size including power calculation assumptions",
          "exposure_assessment": "Detailed description of how exposure will be measured and validated",
          "outcome_assessment": "Detailed description of how outcome will be measured and validated",
          "confounders": ["confounder 1", "confounder 2", "confounder 3"],
          "analysis_plan": "Detailed statistical analysis plan including primary and secondary analyses",
          "limitations": ["limitation 1", "limitation 2", "limitation 3"],
          "ethical_considerations": "Key ethical issues, IRB requirements, informed consent approach",
          "timeline": "Study timeline by phase (e.g. Year 1: recruitment, Year 2-3: follow-up, Year 4: analysis)",
          "resources": "Personnel, equipment, and funding requirements",
          "gap_resolutions": [
            {{"uncertainty": "uncertainty text", "how_addressed": "how this study addresses it"}}
          ],
          "alternative_designs": [
            {{
              "study_type": "Alternative design name",
              "description": "Brief description of this design in context",
              "strengths": ["strength 1", "strength 2"],
              "weaknesses": ["weakness 1", "weakness 2"]
            }}
          ]
        }}

        For gap_resolutions, include one entry for EACH of these uncertainties:
        {json.dumps(top_uncertainties)}

        For alternative_designs, provide 2-3 alternative study designs.

        Use language such as "suggest", "indicate", "consistent with" — never "prove".
    """)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=_FULL_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rstrip("`").strip()

    data = json.loads(raw)

    gap_resolutions = [
        GapResolution(**item) for item in data.get("gap_resolutions", [])
    ]
    # Ensure every top uncertainty has a gap resolution
    resolved_uncertainties = {gr.uncertainty for gr in gap_resolutions}
    for uncertainty in top_uncertainties:
        if uncertainty not in resolved_uncertainties:
            gap_resolutions.append(GapResolution(
                uncertainty=uncertainty,
                how_addressed="Not explicitly addressed by the proposed design.",
            ))

    alternative_designs = [
        AlternativeDesign(**item) for item in data.get("alternative_designs", [])
    ]

    return StudyProposal(
        title=data.get("title", f"Investigation of {request.exposure} and {request.outcome}"),
        study_type=data.get("study_type", "Prospective Cohort Study"),
        hypothesis=data.get("hypothesis", ""),
        population=data.get("population", ""),
        sample_size_rationale=data.get("sample_size_rationale", ""),
        exposure_assessment=data.get("exposure_assessment", ""),
        outcome_assessment=data.get("outcome_assessment", ""),
        confounders=data.get("confounders", []),
        analysis_plan=data.get("analysis_plan", ""),
        limitations=data.get("limitations", []),
        ethical_considerations=data.get("ethical_considerations", ""),
        timeline=data.get("timeline", ""),
        resources=data.get("resources", ""),
        gap_resolutions=gap_resolutions,
        alternative_designs=alternative_designs,
        disclaimer=(
            "Generated proposal \u2014 requires expert review and institutional "
            "adaptation before submission."
        ),
    )


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def render_markdown(proposal: StudyProposal, mode: str) -> str:
    """Render a StudyProposal as clean Markdown.

    Parameters
    ----------
    proposal:
        The StudyProposal to render.
    mode:
        "brief" renders a condensed investigation brief;
        "full" renders a complete academic proposal with all sections.

    Returns
    -------
    str
        Markdown-formatted document ending with a bold disclaimer.
    """
    lines: list[str] = []

    lines.append(f"# {proposal.title}")
    lines.append("")
    lines.append(f"**Study Type:** {proposal.study_type}")
    lines.append("")

    if mode == "brief":
        # --- Finding Summary ---
        lines.append("## Finding Summary")
        lines.append("")
        lines.append(proposal.hypothesis)
        lines.append("")

        # --- Evidence Assessment ---
        lines.append("## Evidence Assessment")
        lines.append("")
        lines.append(f"**Population:** {proposal.population}")
        lines.append("")
        lines.append(f"**Exposure:** {proposal.exposure_assessment}")
        lines.append("")
        lines.append(f"**Outcome:** {proposal.outcome_assessment}")
        lines.append("")

        # --- Recommended Actions ---
        lines.append("## Recommended Actions")
        lines.append("")
        if proposal.analysis_plan:
            lines.append(proposal.analysis_plan)
        lines.append("")

        # --- Limitations ---
        lines.append("## Limitations")
        lines.append("")
        for lim in proposal.limitations:
            lines.append(f"- {lim}")
        lines.append("")

    else:
        # --- Hypothesis ---
        lines.append("## Hypothesis")
        lines.append("")
        lines.append(proposal.hypothesis)
        lines.append("")

        # --- Study Population ---
        lines.append("## Study Population")
        lines.append("")
        lines.append(proposal.population)
        lines.append("")

        # --- Sample Size ---
        lines.append("## Sample Size Rationale")
        lines.append("")
        lines.append(proposal.sample_size_rationale)
        lines.append("")

        # --- Exposure & Outcome Assessment ---
        lines.append("## Exposure and Outcome Assessment")
        lines.append("")
        lines.append(f"**Exposure assessment:** {proposal.exposure_assessment}")
        lines.append("")
        lines.append(f"**Outcome assessment:** {proposal.outcome_assessment}")
        lines.append("")

        # --- Confounders ---
        lines.append("## Confounders to Control")
        lines.append("")
        for conf in proposal.confounders:
            lines.append(f"- {conf}")
        lines.append("")

        # --- Analysis Plan ---
        lines.append("## Analysis Plan")
        lines.append("")
        lines.append(proposal.analysis_plan)
        lines.append("")

        # --- Limitations ---
        lines.append("## Limitations")
        lines.append("")
        for lim in proposal.limitations:
            lines.append(f"- {lim}")
        lines.append("")

        # --- Ethical Considerations ---
        lines.append("## Ethical Considerations")
        lines.append("")
        lines.append(proposal.ethical_considerations)
        lines.append("")

        # --- Timeline & Resources ---
        lines.append("## Timeline")
        lines.append("")
        lines.append(proposal.timeline)
        lines.append("")

        lines.append("## Resources")
        lines.append("")
        lines.append(proposal.resources)
        lines.append("")

        # --- Gap Resolutions Table ---
        if proposal.gap_resolutions:
            lines.append("## Gap Resolution")
            lines.append("")
            lines.append("| Uncertainty | How Addressed |")
            lines.append("|-------------|---------------|")
            for gr in proposal.gap_resolutions:
                uncertainty = gr.uncertainty.replace("|", "\\|")
                how_addressed = gr.how_addressed.replace("|", "\\|")
                lines.append(f"| {uncertainty} | {how_addressed} |")
            lines.append("")

        # --- Alternative Designs Table ---
        if proposal.alternative_designs:
            lines.append("## Alternative Study Designs")
            lines.append("")
            lines.append("| Study Type | Description | Strengths | Weaknesses |")
            lines.append("|------------|-------------|-----------|------------|")
            for alt in proposal.alternative_designs:
                study_type = alt.study_type.replace("|", "\\|")
                description = alt.description.replace("|", "\\|")
                strengths = "; ".join(alt.strengths).replace("|", "\\|")
                weaknesses = "; ".join(alt.weaknesses).replace("|", "\\|")
                lines.append(f"| {study_type} | {description} | {strengths} | {weaknesses} |")
            lines.append("")

    # --- Disclaimer (always bold, always last) ---
    lines.append("---")
    lines.append("")
    lines.append(f"**{proposal.disclaimer}**")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class StudyDesignAgent:
    """Orchestrates study design proposal generation in brief or full mode.

    Parameters
    ----------
    None \u2014 client is passed at run time to match the project pattern.
    """

    def run(
        self,
        request: StudyDesignRequest,
        client: anthropic.Anthropic,
    ) -> StudyDesignResult:
        """Generate a study design proposal and render it as Markdown.

        Parameters
        ----------
        request:
            A fully populated StudyDesignRequest.
        client:
            An initialised Anthropic client.

        Returns
        -------
        StudyDesignResult
            Contains the proposal, rendered markdown, and echoed request.
        """
        if request.output_mode == "brief":
            proposal = generate_investigation_brief(request, client)
        else:
            proposal = generate_full_proposal(request, client)

        md = render_markdown(proposal, request.output_mode)

        return StudyDesignResult(
            request=request,
            proposal=proposal,
            markdown=md,
            mode=request.output_mode,
        )

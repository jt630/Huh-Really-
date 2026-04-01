"""
Tests for src/agents/literature.py

All external I/O (Claude API + PubMed HTTP) is mocked so tests run offline.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.literature import (
    LiteratureAgent,
    LiteratureRequest,
    LiteratureResult,
    LiteratureSynthesis,
    Paper,
    _parse_papers_from_xml,
    generate_pubmed_queries,
    search_pubmed,
    synthesize_with_claude,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

SAMPLE_PUBMED_XML = textwrap.dedent(
    """\
    <?xml version="1.0" encoding="UTF-8"?>
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID Version="1">12345678</PMID>
          <Article>
            <Journal>
              <Title>Journal of Testing</Title>
              <JournalIssue>
                <PubDate><Year>2022</Year></PubDate>
              </JournalIssue>
            </Journal>
            <ArticleTitle>Paraquat and Parkinson Disease Risk</ArticleTitle>
            <Abstract>
              <AbstractText Label="BACKGROUND">Paraquat is a herbicide.</AbstractText>
              <AbstractText Label="RESULTS">Exposure increased risk by 2-fold.</AbstractText>
            </Abstract>
            <AuthorList>
              <Author>
                <LastName>Smith</LastName>
                <ForeName>John</ForeName>
              </Author>
              <Author>
                <LastName>Jones</LastName>
                <ForeName>Alice</ForeName>
              </Author>
            </AuthorList>
          </Article>
          <MeshHeadingList>
            <MeshHeading>
              <DescriptorName>Paraquat</DescriptorName>
            </MeshHeading>
            <MeshHeading>
              <DescriptorName>Parkinson Disease</DescriptorName>
            </MeshHeading>
          </MeshHeadingList>
        </MedlineCitation>
        <PubmedData>
          <ArticleIdList>
            <ArticleId IdType="doi">10.1234/test.2022</ArticleId>
            <ArticleId IdType="pubmed">12345678</ArticleId>
          </ArticleIdList>
        </PubmedData>
      </PubmedArticle>
    </PubmedArticleSet>
    """
)

SAMPLE_SYNTHESIS_JSON = {
    "supporting_count": 1,
    "contradicting_count": 0,
    "evidence_weight": 0.7,
    "key_findings": ["Smith 2022 found 2-fold increased risk"],
    "contradictions": [],
    "dose_response_evidence": "None identified",
    "evidence_gaps": ["Long-term cohort studies needed"],
    "suggested_queries": ['paraquat "dose-response" Parkinson [MeSH Terms]'],
    "confidence": "Moderate",
}

SAMPLE_SYNTHESIS_RESPONSE = (
    "SECTION A \u2014 EVIDENCE\n"
    "- Smith 2022: 'Exposure increased risk by 2-fold.'\n\n"
    "SECTION B \u2014 ANALYSIS\n"
    "[AI ANALYSIS] The single study suggests a positive association.\n\n"
    + json.dumps(SAMPLE_SYNTHESIS_JSON)
)


def _make_claude_message(text: str) -> MagicMock:
    content_block = MagicMock()
    content_block.text = text
    message = MagicMock()
    message.content = [content_block]
    return message


def _make_mock_client(responses: list[str]) -> MagicMock:
    client = MagicMock()
    client.messages.create.side_effect = [_make_claude_message(r) for r in responses]
    return client


class TestParsePapersFromXml:
    def test_parses_single_article(self) -> None:
        papers = _parse_papers_from_xml(SAMPLE_PUBMED_XML)
        assert len(papers) == 1
        p = papers[0]
        assert p.pmid == "12345678"
        assert "Paraquat" in p.title
        assert "BACKGROUND" in p.abstract
        assert "RESULTS" in p.abstract
        assert p.authors == ["Smith John", "Jones Alice"]
        assert p.journal == "Journal of Testing"
        assert p.year == "2022"
        assert "Paraquat" in p.mesh_terms
        assert "Parkinson Disease" in p.mesh_terms
        assert p.doi == "10.1234/test.2022"

    def test_empty_xml_returns_empty_list(self) -> None:
        xml = '<?xml version="1.0"?><PubmedArticleSet></PubmedArticleSet>'
        assert _parse_papers_from_xml(xml) == []


class TestGeneratePubmedQueries:
    @patch("src.agents.literature.get_settings")
    def test_returns_list_of_strings(self, mock_settings: MagicMock) -> None:
        mock_settings.return_value.claude_model = "claude-sonnet-4-6"
        queries_json = json.dumps(
            [
                "paraquat [MeSH Terms] AND Parkinson Disease [MeSH Terms]",
                "herbicide exposure AND neurodegenerative disease",
                "paraquat oxidative stress dopaminergic neurons",
            ]
        )
        client = _make_mock_client([queries_json])
        result = generate_pubmed_queries("Paraquat causes Parkinson disease", client)
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(q, str) for q in result)

    @patch("src.agents.literature.get_settings")
    def test_strips_markdown_fences(self, mock_settings: MagicMock) -> None:
        mock_settings.return_value.claude_model = "claude-sonnet-4-6"
        fenced = '```json\n["query A", "query B"]\n```'
        client = _make_mock_client([fenced])
        result = generate_pubmed_queries("test hypothesis", client)
        assert result == ["query A", "query B"]

    @patch("src.agents.literature.get_settings")
    def test_claude_called_with_hypothesis(self, mock_settings: MagicMock) -> None:
        mock_settings.return_value.claude_model = "claude-sonnet-4-6"
        client = _make_mock_client(['["q1"]'])
        generate_pubmed_queries("my specific hypothesis", client)
        call_kwargs = client.messages.create.call_args
        prompt_text = call_kwargs[1]["messages"][0]["content"]
        assert "my specific hypothesis" in prompt_text


class TestSearchPubmed:
    @patch("src.agents.literature.get_settings")
    @patch("src.agents.literature._efetch_xml")
    @patch("src.agents.literature._esearch")
    @patch("src.agents.literature.time.sleep")
    def test_returns_papers(self, mock_sleep, mock_esearch, mock_efetch, mock_settings, tmp_path):
        mock_settings.return_value.cache_dir = tmp_path
        mock_esearch.return_value = ["12345678"]
        mock_efetch.return_value = SAMPLE_PUBMED_XML
        papers = search_pubmed(["paraquat parkinson"], max_results=10, api_key=None)
        assert len(papers) == 1
        assert papers[0].pmid == "12345678"

    @patch("src.agents.literature.get_settings")
    @patch("src.agents.literature._efetch_xml")
    @patch("src.agents.literature._esearch")
    @patch("src.agents.literature.time.sleep")
    def test_deduplicates_across_queries(self, mock_sleep, mock_esearch, mock_efetch, mock_settings, tmp_path):
        mock_settings.return_value.cache_dir = tmp_path
        mock_esearch.return_value = ["12345678"]
        mock_efetch.return_value = SAMPLE_PUBMED_XML
        papers = search_pubmed(["query1", "query2"], max_results=10, api_key=None)
        assert len(papers) == 1

    @patch("src.agents.literature.get_settings")
    @patch("src.agents.literature._efetch_xml")
    @patch("src.agents.literature._esearch")
    @patch("src.agents.literature.time.sleep")
    def test_uses_cache_on_second_call(self, mock_sleep, mock_esearch, mock_efetch, mock_settings, tmp_path):
        mock_settings.return_value.cache_dir = tmp_path
        mock_esearch.return_value = ["12345678"]
        mock_efetch.return_value = SAMPLE_PUBMED_XML
        search_pubmed(["cached query"], max_results=10, api_key=None)
        search_pubmed(["cached query"], max_results=10, api_key=None)
        assert mock_efetch.call_count == 1

    @patch("src.agents.literature.get_settings")
    @patch("src.agents.literature._esearch")
    @patch("src.agents.literature.time.sleep")
    def test_skips_query_with_no_pmids(self, mock_sleep, mock_esearch, mock_settings, tmp_path):
        mock_settings.return_value.cache_dir = tmp_path
        mock_esearch.return_value = []
        papers = search_pubmed(["obscure query"], max_results=10, api_key=None)
        assert papers == []


class TestSynthesizeWithClaude:
    @patch("src.agents.literature.get_settings")
    def test_returns_synthesis(self, mock_settings: MagicMock) -> None:
        mock_settings.return_value.claude_model = "claude-sonnet-4-6"
        client = _make_mock_client([SAMPLE_SYNTHESIS_RESPONSE])
        papers = _parse_papers_from_xml(SAMPLE_PUBMED_XML)
        result = synthesize_with_claude("paraquat causes Parkinson", papers, client)
        assert isinstance(result, LiteratureSynthesis)
        assert result.supporting_count == 1
        assert result.contradicting_count == 0
        assert result.evidence_weight == pytest.approx(0.7)
        assert result.confidence == "Moderate"
        assert len(result.key_findings) >= 1

    @patch("src.agents.literature.get_settings")
    def test_raises_on_missing_json(self, mock_settings: MagicMock) -> None:
        mock_settings.return_value.claude_model = "claude-sonnet-4-6"
        client = _make_mock_client(["No JSON here at all."])
        papers = _parse_papers_from_xml(SAMPLE_PUBMED_XML)
        with pytest.raises(ValueError, match="No JSON found"):
            synthesize_with_claude("hypothesis", papers, client)

    @patch("src.agents.literature.get_settings")
    def test_handles_empty_paper_list(self, mock_settings: MagicMock) -> None:
        mock_settings.return_value.claude_model = "claude-sonnet-4-6"
        empty_json = json.dumps({
            "supporting_count": 0, "contradicting_count": 0, "evidence_weight": 0.0,
            "key_findings": [], "contradictions": [], "dose_response_evidence": "None identified",
            "evidence_gaps": ["No studies found"], "suggested_queries": [], "confidence": "Insufficient data",
        })
        client = _make_mock_client([f"SECTION A\n(none)\nSECTION B\n(none)\n{empty_json}"])
        result = synthesize_with_claude("obscure hypothesis", [], client)
        assert result.supporting_count == 0
        assert result.confidence == "Insufficient data"


class TestLiteratureAgentRun:
    def _make_agent(self, claude_responses: list[str]) -> LiteratureAgent:
        agent = object.__new__(LiteratureAgent)
        agent.client = _make_mock_client(claude_responses)
        agent.ncbi_api_key = None
        return agent

    @patch("src.agents.literature.search_pubmed")
    def test_depth_1_basic(self, mock_search: MagicMock) -> None:
        mock_search.return_value = _parse_papers_from_xml(SAMPLE_PUBMED_XML)
        queries_json = '["query1", "query2", "query3"]'
        agent = self._make_agent([queries_json, SAMPLE_SYNTHESIS_RESPONSE])
        request = LiteratureRequest(hypothesis="paraquat causes Parkinson", depth=1)
        result = agent.run(request)
        assert isinstance(result, LiteratureResult)
        assert result.queries_used == ["query1", "query2", "query3"]
        assert len(result.papers) == 1
        assert result.synthesis.supporting_count == 1

    @patch("src.agents.literature.search_pubmed")
    def test_result_has_correct_shape(self, mock_search: MagicMock) -> None:
        mock_search.return_value = []
        empty_synthesis_json = json.dumps({
            "supporting_count": 0, "contradicting_count": 0, "evidence_weight": 0.0,
            "key_findings": [], "contradictions": [], "dose_response_evidence": "None identified",
            "evidence_gaps": [], "suggested_queries": [], "confidence": "Insufficient data",
        })
        agent = self._make_agent(['["q1"]', f"SECTION A\nnone\nSECTION B\nnone\n{empty_synthesis_json}"])
        request = LiteratureRequest(hypothesis="test", max_results=10, depth=1)
        result = agent.run(request)
        assert result.request == request
        assert result.papers == []
        assert result.queries_used == ["q1"]
        assert isinstance(result.synthesis, LiteratureSynthesis)

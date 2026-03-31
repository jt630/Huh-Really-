"""
Pipeline orchestrator - to be implemented in Prompt 7.
"""
from __future__ import annotations


class Pipeline:
    def run_hypothesis(self, exposure: str, outcome: str, confounders: list[str], output_mode: str = "brief") -> "PipelineResult":
        raise NotImplementedError

    def run_discovery(self, outcome: str, top_n: int = 20, confounders: list[str] | None = None, output_mode: str = "brief") -> list["PipelineResult"]:
        raise NotImplementedError


class PipelineResult:
    pass


def main() -> None:
    raise NotImplementedError

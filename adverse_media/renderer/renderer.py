"""Renders a ScreeningResult as a formatted Rich report to the terminal."""

from __future__ import annotations

import textwrap
from typing import List, Optional

from rich import box
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..checker.models import ScreeningResult, StatisticalScreeningResult

_VERDICT_STYLE = {
    "DISCARD": "bold green",
    "POSSIBLE_MATCH": "bold yellow",
    "LIKELY_MATCH": "bold red",
}
_SENTIMENT_STYLE = {
    "POSITIVE": "bold green",
    "NEGATIVE": "bold red",
    "NEUTRAL": "bold blue",
    "MIXED": "bold yellow",
}


class Renderer:
    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _pct(value: Optional[float]) -> str:
        return f"{value * 100:.0f}%" if value is not None else "—"

    # ── Panel / text builders ─────────────────────────────────────────────────

    @staticmethod
    def build_query_panel(name: str, dob: Optional[str], url: str, language: str) -> Panel:
        qt = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        qt.add_column("k", style="bold cyan", width=18)
        qt.add_column("v")
        qt.add_row("Query name", escape(name))
        qt.add_row("Date of birth", escape(dob or "not provided"))
        qt.add_row("Article URL", escape((url[:80] + "…") if len(url) > 80 else url))
        qt.add_row("Article language", escape(language))
        return Panel(qt, title="Query", border_style="cyan")

    @staticmethod
    def build_persons_panel(persons_found: List[dict]) -> Optional[Panel]:
        if not persons_found:
            return None
        pt = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        pt.add_column("Name in article", style="bold")
        pt.add_column("Context / role")
        for p in persons_found:
            pt.add_row(
                escape(str(p.get("name_in_article", ""))),
                escape(str(p.get("role", ""))),
            )
        return Panel(pt, title="Named individuals found", border_style="dim white")

    @staticmethod
    def build_verdict_text(
        match_assessment: str,
        match_confidence: Optional[float],
        matched_name_in_article: Optional[str],
    ) -> Text:
        verdict_style = _VERDICT_STYLE.get(match_assessment, "bold white")
        vt = Text()
        vt.append(f"  VERDICT: {match_assessment}  ", style=f"{verdict_style} reverse")
        vt.append(f"   confidence: {Renderer._pct(match_confidence)}")
        if matched_name_in_article:
            vt.append(f'   matched as: "{escape(matched_name_in_article)}"')
        return vt

    @staticmethod
    def build_match_reasoning_panel(reasoning: Optional[str], border_color: str) -> Optional[Panel]:
        if not reasoning:
            return None
        return Panel(
            textwrap.fill(reasoning, width=88),
            title="[bold]Match reasoning[/bold]",
            border_style=border_color,
        )

    @staticmethod
    def build_sentiment_text(sentiment: str, sentiment_confidence: Optional[float]) -> Text:
        sent_style = _SENTIMENT_STYLE.get(sentiment, "bold white")
        st = Text()
        st.append("  Sentiment: ", style="bold")
        st.append(f" {sentiment} ", style=f"{sent_style} reverse")
        st.append(f"   confidence: {Renderer._pct(sentiment_confidence)}")
        return st

    @staticmethod
    def build_sentiment_reasoning_panel(reasoning: Optional[str], border_color: str) -> Optional[Panel]:
        if not reasoning:
            return None
        return Panel(
            textwrap.fill(reasoning, width=88),
            title="[bold]Sentiment reasoning[/bold]",
            border_style=border_color,
        )

    @staticmethod
    def build_statistical_panel(stat: StatisticalScreeningResult) -> Panel:
        border_color = "red" if stat.has_adverse_signal else "green"
        signal_label = "[bold red]YES[/bold red]" if stat.has_adverse_signal else "[bold green]NO[/bold green]"

        st = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        st.add_column("k", style="bold", width=20)
        st.add_column("v")
        st.add_row("Adverse signal", signal_label)
        st.add_row("Risk score", Renderer._pct(stat.risk_score))

        if stat.adverse_entity_hits:
            ht = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
            ht.add_column("Entity", style="bold")
            ht.add_column("Linked adverse signals")
            for entity, lemmas in stat.adverse_entity_hits.items():
                ht.add_row(escape(entity), escape(", ".join(lemmas)))

            content: object = Group(st, ht)
        else:
            content = st

        return Panel(content, title="Statistical pre-screen", border_style=border_color)

    @staticmethod
    def build_analyst_note_panel(analyst_note: Optional[str]) -> Optional[Panel]:
        if not analyst_note:
            return None
        return Panel(
            textwrap.fill(analyst_note, width=88),
            title="[bold yellow]Analyst note[/bold yellow]",
            border_style="yellow",
        )

    # ── Main entry point ──────────────────────────────────────────────────────

    def render(self, name: str, dob: Optional[str], url: str, result: ScreeningResult) -> None:
        """Print a formatted screening report to the terminal."""
        verdict_style = _VERDICT_STYLE.get(result.match_assessment, "bold white")
        border_color = verdict_style.split()[-1]

        self.console.print()
        self.console.rule("[bold]Adverse Media Screening Report[/bold]")
        self.console.print()

        self.console.print(self.build_query_panel(name, dob, url, result.language))

        persons_panel = self.build_persons_panel(result.persons_found or [])
        if persons_panel is not None:
            self.console.print(persons_panel)

        self.console.print()
        self.console.print(self.build_verdict_text(
            result.match_assessment, result.match_confidence, result.matched_name_in_article
        ))
        self.console.print()

        match_panel = self.build_match_reasoning_panel(result.match_reasoning, border_color)
        if match_panel is not None:
            self.console.print(match_panel)

        if result.statistical_result is not None:
            self.console.print(self.build_statistical_panel(result.statistical_result))

        if result.dob_evidence and result.dob_evidence.lower() not in ("none found", "none", ""):
            self.console.print(f"\n  [bold]DOB / age evidence:[/bold] {escape(result.dob_evidence)}")

        if result.match_assessment != "DISCARD" and result.sentiment:
            sent_style = _SENTIMENT_STYLE.get(result.sentiment, "bold white")
            sent_border = sent_style.split()[-1]

            self.console.print()
            self.console.print(self.build_sentiment_text(result.sentiment, result.sentiment_confidence))

            sent_panel = self.build_sentiment_reasoning_panel(result.sentiment_reasoning, sent_border)
            if sent_panel is not None:
                self.console.print(sent_panel)

            if result.key_adverse_facts:
                self.console.print("\n  [bold red]Key adverse facts:[/bold red]")
                for fact in result.key_adverse_facts:
                    self.console.print(f"    • {escape(str(fact))}")

            if result.key_positive_facts:
                self.console.print("\n  [bold green]Key positive facts:[/bold green]")
                for fact in result.key_positive_facts:
                    self.console.print(f"    • {escape(str(fact))}")

        analyst_panel = self.build_analyst_note_panel(result.analyst_note)
        if analyst_panel is not None:
            self.console.print()
            self.console.print(analyst_panel)

        self.console.print()
        self.console.rule()
        self.console.print()


renderer = Renderer()

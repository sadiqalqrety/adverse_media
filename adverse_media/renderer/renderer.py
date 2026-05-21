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
    def build_persons_panel(persons_found: List[dict]) -> Panel:
        if not persons_found:
            return Panel("No individuals identified.", title="Named individuals found", border_style="dim white")
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
    def build_match_reasoning_panel(reasoning: Optional[str], border_color: str) -> Panel:
        return Panel(
            textwrap.fill(reasoning, width=88) if reasoning else "No match reasoning provided.",
            title="[bold]Match reasoning[/bold]",
            border_style=border_color,
        )

    @staticmethod
    def build_statistical_panel(stat: Optional[StatisticalScreeningResult]) -> Panel:
        if stat is None:
            return Panel("No statistical result.", title="Statistical pre-screen", border_style="dim white")

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
    def build_sentiment_text(sentiment: str, sentiment_confidence: Optional[float]) -> Text:
        sent_style = _SENTIMENT_STYLE.get(sentiment, "bold white")
        st = Text()
        st.append("  Sentiment: ", style="bold")
        st.append(f" {sentiment} ", style=f"{sent_style} reverse")
        st.append(f"   confidence: {Renderer._pct(sentiment_confidence)}")
        return st

    @staticmethod
    def build_sentiment_reasoning_panel(reasoning: Optional[str], border_color: str) -> Panel:
        return Panel(
            textwrap.fill(reasoning, width=88) if reasoning else "No sentiment reasoning provided.",
            title="[bold]Sentiment reasoning[/bold]",
            border_style=border_color,
        )

    @staticmethod
    def build_analyst_note_panel(analyst_note: Optional[str]) -> Panel:
        return Panel(
            textwrap.fill(analyst_note, width=88) if analyst_note else "No analyst note.",
            title="[bold yellow]Analyst note[/bold yellow]",
            border_style="yellow",
        )

    # ── Brief output (default) ────────────────────────────────────────────────

    def _render_brief(self, result: ScreeningResult) -> None:
        verdict_style = _VERDICT_STYLE.get(result.match_assessment, "bold white")
        line1 = Text()
        line1.append(f"  {result.match_assessment}  ", style=f"{verdict_style} reverse")
        line1.append(f"   {self._pct(result.match_confidence)} confidence")
        if result.matched_name_in_article:
            line1.append(f'   matched as: "{escape(result.matched_name_in_article)}"')

        self.console.print()
        self.console.print(line1)

        if result.match_assessment != "DISCARD" and result.sentiment:
            self.console.print(self.build_sentiment_text(result.sentiment, result.sentiment_confidence))

        self.console.print()

    # ── Main entry point ──────────────────────────────────────────────────────

    def render(
        self,
        name: str,
        dob: Optional[str],
        url: str,
        result: ScreeningResult,
        *,
        detailed: bool = False,
    ) -> None:
        """Print a screening report to the terminal.

        By default prints two lines: match verdict and sentiment.
        Pass detailed=True for the full structured report.
        """
        if not detailed:
            self._render_brief(result)
            return

        verdict_style = _VERDICT_STYLE.get(result.match_assessment, "bold white")
        border_color = verdict_style.split()[-1]

        self.console.print()
        self.console.rule("[bold]Adverse Media Screening Report[/bold]")
        self.console.print()

        self.console.print(self.build_query_panel(name, dob, url, result.language))
        self.console.print(self.build_persons_panel(result.persons_found or []))

        self.console.print()
        self.console.print(self.build_verdict_text(
            result.match_assessment, result.match_confidence, result.matched_name_in_article
        ))
        self.console.print()

        self.console.print(self.build_match_reasoning_panel(result.match_reasoning, border_color))
        self.console.print(self.build_statistical_panel(result.statistical_result))

        if result.dob_evidence and result.dob_evidence.lower() not in ("none found", "none", ""):
            self.console.print(f"\n  [bold]DOB / age evidence:[/bold] {escape(result.dob_evidence)}")

        if result.match_assessment != "DISCARD" and result.sentiment:
            sent_style = _SENTIMENT_STYLE.get(result.sentiment, "bold white")
            sent_border = sent_style.split()[-1]

            self.console.print()
            self.console.print(self.build_sentiment_text(result.sentiment, result.sentiment_confidence))
            self.console.print(self.build_sentiment_reasoning_panel(result.sentiment_reasoning, sent_border))

            if result.key_adverse_facts:
                self.console.print("\n  [bold red]Key adverse facts:[/bold red]")
                for fact in result.key_adverse_facts:
                    self.console.print(f"    • {escape(str(fact))}")

            if result.key_positive_facts:
                self.console.print("\n  [bold green]Key positive facts:[/bold green]")
                for fact in result.key_positive_facts:
                    self.console.print(f"    • {escape(str(fact))}")

        self.console.print()
        self.console.print(self.build_analyst_note_panel(result.analyst_note))

        self.console.print()
        self.console.rule()
        self.console.print()


renderer = Renderer()

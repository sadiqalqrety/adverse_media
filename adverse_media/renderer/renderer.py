"""Renders a ScreeningResult as a formatted Rich report to the terminal."""

from __future__ import annotations

import textwrap
from typing import Optional

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..checker.models import ScreeningResult

console = Console()

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


def _pct(value: Optional[float]) -> str:
    return f"{value * 100:.0f}%" if value is not None else "—"


def render(name: str, dob: Optional[str], url: str, result: ScreeningResult) -> None:
    """Print a formatted screening report to the terminal."""
    verdict_style = _VERDICT_STYLE.get(result.match_assessment, "bold white")
    border_color = verdict_style.split()[-1]

    console.print()
    console.rule("[bold]Adverse Media Screening Report[/bold]")
    console.print()

    # ── Query summary ─────────────────────────────────────────────────────────
    qt = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    qt.add_column("k", style="bold cyan", width=18)
    qt.add_column("v")
    qt.add_row("Query name", escape(name))
    qt.add_row("Date of birth", escape(dob or "not provided"))
    qt.add_row("Article URL", escape((url[:80] + "…") if len(url) > 80 else url))
    qt.add_row("Article language", escape(result.language))
    console.print(Panel(qt, title="Query", border_style="cyan"))

    # ── Persons found ─────────────────────────────────────────────────────────
    if result.persons_found:
        pt = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        pt.add_column("Name in article", style="bold")
        pt.add_column("Context / role")
        for p in result.persons_found:
            pt.add_row(
                escape(str(p.get("name_in_article", ""))),
                escape(str(p.get("role", ""))),
            )
        console.print(Panel(pt, title="Named individuals found", border_style="dim white"))

    # ── Match verdict ─────────────────────────────────────────────────────────
    console.print()
    vt = Text()
    vt.append(f"  VERDICT: {result.match_assessment}  ", style=f"{verdict_style} reverse")
    vt.append(f"   confidence: {_pct(result.match_confidence)}")
    if result.matched_name_in_article:
        vt.append(f'   matched as: "{escape(result.matched_name_in_article)}"')
    console.print(vt)
    console.print()

    if result.match_reasoning:
        console.print(Panel(
            textwrap.fill(result.match_reasoning, width=88),
            title="[bold]Match reasoning[/bold]",
            border_style=border_color,
        ))

    if result.dob_evidence and result.dob_evidence.lower() not in ("none found", "none", ""):
        console.print(f"\n  [bold]DOB / age evidence:[/bold] {escape(result.dob_evidence)}")

    # ── Sentiment ─────────────────────────────────────────────────────────────
    if result.match_assessment != "DISCARD" and result.sentiment:
        sent_style = _SENTIMENT_STYLE.get(result.sentiment, "bold white")
        sent_border = sent_style.split()[-1]

        console.print()
        st = Text()
        st.append("  Sentiment: ", style="bold")
        st.append(f" {result.sentiment} ", style=f"{sent_style} reverse")
        st.append(f"   confidence: {_pct(result.sentiment_confidence)}")
        console.print(st)

        if result.sentiment_reasoning:
            console.print(Panel(
                textwrap.fill(result.sentiment_reasoning, width=88),
                title="[bold]Sentiment reasoning[/bold]",
                border_style=sent_border,
            ))

        if result.key_adverse_facts:
            console.print("\n  [bold red]Key adverse facts:[/bold red]")
            for fact in result.key_adverse_facts:
                console.print(f"    • {escape(str(fact))}")

        if result.key_positive_facts:
            console.print("\n  [bold green]Key positive facts:[/bold green]")
            for fact in result.key_positive_facts:
                console.print(f"    • {escape(str(fact))}")

    # ── Analyst note ──────────────────────────────────────────────────────────
    if result.analyst_note:
        console.print()
        console.print(Panel(
            textwrap.fill(result.analyst_note, width=88),
            title="[bold yellow]Analyst note[/bold yellow]",
            border_style="yellow",
        ))

    console.print()
    console.rule()
    console.print()

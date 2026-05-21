"""CLI helper functions: argument parser, interactive prompt, DOB normalisation."""

from __future__ import annotations

import argparse
import textwrap
from datetime import datetime


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m adverse_media.app",
        description=(
            "Screen a news article for mentions of a specific individual "
            "and classify any coverage as positive or negative."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              # Fully interactive — prompts for all inputs:
              python -m adverse_media.app

              # All arguments inline:
              python -m adverse_media.app --name "James Smith" --dob 1985-03-15 --url "https://..."

              # Without DOB (reduced age-corroboration accuracy):
              python -m adverse_media.app --name "Wei Li" --url "https://..."

              # Machine-readable JSON for downstream pipelines:
              python -m adverse_media.app --name "..." --url "..." --json

              # Full structured report (default is a two-line summary):
              python -m adverse_media.app --name "James Smith" --url "https://..." --detailed True

              # Skip the Claude LLM call — statistical pre-screen only (no API key required):
              python -m adverse_media.app --name "James Smith" --url "https://..." --skip-llm-semantic-extractor True
        """),
    )
    p.add_argument("--name", help="Full name of the individual to screen for")
    p.add_argument(
        "--dob", metavar="YYYY-MM-DD",
        help="Date of birth — ISO 8601 preferred; DD/MM/YYYY also accepted",
    )
    p.add_argument("--url", help="URL of the news article to analyse")
    p.add_argument(
        "--json", action="store_true", dest="output_json",
        help="Print raw JSON instead of a formatted report",
    )
    p.add_argument(
        "--detailed", type=lambda x: x.lower() == "true", default=False,
        metavar="True|False",
        help="Print the full structured report (default: False — two-line summary)",
    )
    p.add_argument(
        "--skip-llm-semantic-extractor",
        type=lambda x: x.lower() == "true", default=False,
        dest="skip_llm_semantic_extractor",
        metavar="True|False",
        help=(
            "Skip the Claude LLM call and return statistical pre-screen results only "
            "(default: False). No ANTHROPIC_API_KEY required when True."
        ),
    )
    return p


def prompt(label: str, required: bool = True) -> str:
    """Prompt the user interactively; re-prompt until a value is given if required."""
    while True:
        val = input(f"  {label}: ").strip()
        if val or not required:
            return val
        print("  (required — please enter a value)")


def normalise_dob(raw: str) -> str:
    """Parse several common date formats and return YYYY-MM-DD.

    Raises ValueError if the input cannot be parsed.
    """
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date format: {raw!r} — please use YYYY-MM-DD.")

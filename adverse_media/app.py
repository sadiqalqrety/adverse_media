#!/usr/bin/env python3
"""Adverse Media Screening Tool — entry point.

See README.md for installation instructions and usage examples.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from typing import Optional

from rich.console import Console

from .checker import AdverseMediaChecker
from .cli import build_parser, normalise_dob, prompt
from .logging_config import configure_logging
from .renderer import renderer

console = Console()


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()

    if not args.name or not args.url:
        console.print("\n[bold cyan]━━ Adverse Media Screening Tool ━━[/bold cyan]")
        console.print("Fill in the details below (DOB optional but improves accuracy).\n")

    name: str = args.name or prompt("Full name")
    dob_raw: str = args.dob or prompt("Date of birth (YYYY-MM-DD, optional)", required=False)
    url: str = args.url or prompt("Article URL")

    dob: Optional[str] = None
    if dob_raw:
        try:
            dob = normalise_dob(dob_raw)
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            sys.exit(1)

    checker = AdverseMediaChecker()

    with console.status("[cyan]Fetching and analysing article…[/cyan]"):
        try:
            result = checker.screen(name, dob, url)
        except RuntimeError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            sys.exit(1)

    if args.output_json:
        print(json.dumps(dataclasses.asdict(result), indent=2, ensure_ascii=False))
    else:
        renderer.render(name, dob, url, result, detailed=args.detailed)


if __name__ == "__main__":
    main()

"""CLI entrypoint for LegalIntern."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .core.config import Config
from .engine import LegalIntern


def cli() -> None:
    parser = argparse.ArgumentParser(description="LegalIntern -- multi-agent legal consultation")
    parser.add_argument("question", nargs="?", help="Legal question (or path to .yaml problem file)")
    parser.add_argument("--model", default=None, help="Override default LLM model")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--config", type=Path, default=None, help="Config YAML file")
    parser.add_argument("--workspace-dir", type=str, default=None)
    args = parser.parse_args()

    if not args.question:
        parser.print_help()
        sys.exit(1)

    # Load config
    if args.config and args.config.exists():
        config = Config.from_yaml(args.config)
    else:
        config = Config.from_env()

    if args.model:
        config.default_model = args.model
    if args.max_iterations:
        config.max_iterations = args.max_iterations
    if args.workspace_dir:
        config.workspace_dir = args.workspace_dir

    # Load question from file or use directly
    question = args.question
    if Path(question).exists():
        import yaml

        data = yaml.safe_load(Path(question).read_text())
        question = data.get("question", data.get("problem", question))

    intern = LegalIntern(question, config)
    answer = asyncio.run(intern.run())

    print("\n" + "=" * 60)
    print(answer)


if __name__ == "__main__":
    cli()

"""Command-line interface for workload-guesser.

Usage examples
--------------
Train and run interactively::

    python -m workload_guesser.cli

Predict from command-line flags::

    python -m workload_guesser.cli predict \\
        --department CS \\
        --level 4000 \\
        --credits 3 \\
        --description "Weekly problem sets, two midterms and a final exam."

Save the trained model to disk::

    python -m workload_guesser.cli train --save models/workload.pkl

Load a saved model and predict::

    python -m workload_guesser.cli predict \\
        --model models/workload.pkl \\
        --department MATH --level 4000 --credits 3 \\
        --description "Rigorous proof-based course with weekly homework."
"""

from __future__ import annotations

import argparse
import sys

from workload_guesser.data import course_to_dataframe
from workload_guesser.model import WorkloadPredictor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WORKLOAD_EMOJI = {"low": "🟢", "medium": "🟡", "high": "🔴"}


def _print_prediction(label: str, proba: dict[str, float]) -> None:
    emoji = _WORKLOAD_EMOJI.get(label, "")
    print(f"\nPredicted workload: {emoji}  {label.upper()}\n")
    print("Confidence breakdown:")
    for cat in ["low", "medium", "high"]:
        bar_len = int(proba.get(cat, 0.0) * 30)
        bar = "█" * bar_len
        print(f"  {cat:6s}  {bar:<30s}  {proba.get(cat, 0.0):.1%}")
    print()


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> None:
    """Train the model (optionally saving it) and report CV accuracy."""
    predictor = WorkloadPredictor()
    predictor.train(data_path=args.data, cv=True)
    print("Training complete.")
    if args.save:
        predictor.save(args.save)
        print(f"Model saved to {args.save}")


def cmd_predict(args: argparse.Namespace) -> None:
    """Predict workload for a single course supplied via CLI flags."""
    if args.model:
        predictor = WorkloadPredictor.load(args.model)
    else:
        print("No saved model specified – training on built-in data...")
        predictor = WorkloadPredictor()
        predictor.train()

    df = course_to_dataframe(
        department=args.department,
        level=args.level,
        credits=args.credits,
        description=args.description,
        title=args.title or "",
        gpa_avg=args.gpa_avg,
    )

    label = predictor.predict(df)[0]
    proba = predictor.predict_proba(df)
    _print_prediction(label, proba)


def cmd_interactive(args: argparse.Namespace) -> None:  # noqa: ARG001
    """Interactive prompt-based prediction loop."""
    print("=== Workload Guesser – Interactive Mode ===")
    print("Training model on built-in sample data...")
    predictor = WorkloadPredictor()
    predictor.train()
    print("Ready.\n")

    while True:
        print("Enter course details (or 'quit' to exit):")
        try:
            dept = input("  Department (e.g. CS, MATH): ").strip()
            if dept.lower() in {"quit", "q", "exit"}:
                break

            level_str = input("  Course level (e.g. 1000, 3000): ").strip()
            credits_str = input("  Credits (e.g. 3): ").strip()
            description = input("  Course description: ").strip()
            gpa_str = input("  Average GPA [leave blank to skip]: ").strip()

            level = int(level_str)
            credits_ = int(credits_str)
            gpa_avg = float(gpa_str) if gpa_str else None

        except (ValueError, EOFError) as exc:
            print(f"  Invalid input: {exc}. Please try again.\n")
            continue
        except KeyboardInterrupt:
            print()
            break

        df = course_to_dataframe(
            department=dept,
            level=level,
            credits=credits_,
            description=description,
            gpa_avg=gpa_avg,
        )
        label = predictor.predict(df)[0]
        proba = predictor.predict_proba(df)
        _print_prediction(label, proba)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="workload-guesser",
        description="Predict the workload level (low / medium / high) of a UVA course.",
    )
    sub = parser.add_subparsers(dest="command")

    # --- train sub-command ---
    train_p = sub.add_parser("train", help="Train the model and optionally save it.")
    train_p.add_argument("--data", metavar="CSV", help="Path to training CSV file.")
    train_p.add_argument("--save", metavar="PKL", help="Save trained model to this path.")
    train_p.set_defaults(func=cmd_train)

    # --- predict sub-command ---
    pred_p = sub.add_parser("predict", help="Predict workload for a single course.")
    pred_p.add_argument("--model", metavar="PKL", help="Path to a saved model file.")
    pred_p.add_argument("--department", required=True, help="Department code, e.g. CS.")
    pred_p.add_argument("--level", required=True, type=int, help="Course level, e.g. 4000.")
    pred_p.add_argument("--credits", required=True, type=int, help="Credit hours.")
    pred_p.add_argument("--description", required=True, help="Free-text course description.")
    pred_p.add_argument("--title", default="", help="Course title (optional).")
    pred_p.add_argument(
        "--gpa-avg",
        dest="gpa_avg",
        type=float,
        default=None,
        help="Historical average GPA (optional).",
    )
    pred_p.set_defaults(func=cmd_predict)

    # --- interactive sub-command (default) ---
    int_p = sub.add_parser("interactive", help="Start an interactive prompt session.")
    int_p.set_defaults(func=cmd_interactive)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        # Default to interactive mode when no sub-command is given.
        cmd_interactive(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()

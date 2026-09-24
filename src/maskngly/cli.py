from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from .core import InfluenceModel, read_fasta, site_scores, window


def generate(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    count = 0
    model: InfluenceModel | None = None
    for protein in read_fasta(args.fasta):
        if len(protein.sequence) <= 1022:
            targets = [(1, protein.sequence)]
        else:
            targets = list(
                dict.fromkeys(window(protein.sequence, site) for site in protein.sites)
            )
        for start, sequence in targets:
            name = (
                protein.identifier
                if start == 1 and len(protein.sequence) <= 1022
                else f"{protein.identifier}_{start}"
            )
            path = args.output / f"{name}.npy"
            if path.exists() and not args.overwrite:
                print(f"skip {path}")
                continue
            if model is None:
                model = InfluenceModel(args.device, args.batch_size)
            np.save(path, model.matrix(sequence))
            print(f"wrote {path} ({len(sequence)} residues)")
            count += 1
    print(f"Generated {count} matrices")


def _matrix_for_site(
    directory: Path, identifier: str, position: int
) -> tuple[Path, int]:
    full = directory / f"{identifier}.npy"
    if full.exists():
        return full, position
    choices: list[tuple[int, Path]] = []
    for path in directory.glob(f"{identifier}_*.npy"):
        try:
            start = int(path.stem.removeprefix(f"{identifier}_"))
        except ValueError:
            continue
        matrix = np.load(path, mmap_mode="r")
        if start <= position and position + 2 < start + len(matrix):
            choices.append((start, path))
    if not choices:
        raise FileNotFoundError(f"No matrix contains {identifier} site {position}")
    start, path = min(choices)
    return path, position - start + 1


def score(args: argparse.Namespace) -> None:
    rows: list[dict[str, str | int | float]] = []
    for protein in read_fasta(args.fasta):
        for position in protein.sites:
            path, local_position = _matrix_for_site(
                args.matrices, protein.identifier, position
            )
            original, paired = site_scores(np.load(path), local_position)
            value = original if args.mode == "st" else paired
            rows.append(
                {
                    "protein_id": protein.identifier,
                    "position": position,
                    "motif": protein.sequence[position - 1 : position + 2],
                    "score": value,
                    "prediction": int(value > args.threshold),
                    "matrix": path.name,
                }
            )
    _write_csv(
        args.output,
        rows,
        ["protein_id", "position", "motif", "score", "prediction", "matrix"],
    )
    print(f"Wrote {len(rows)} site scores to {args.output}")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> None:
    try:
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            balanced_accuracy_score,
            confusion_matrix,
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
            roc_auc_score,
            roc_curve,
        )
    except ImportError as error:
        raise RuntimeError(
            "Install the evaluation extra: uv sync --extra evaluation"
        ) from error
    with args.labels.open(newline="", encoding="utf-8") as handle:
        labels = {
            (row["protein_id"], int(row["position"])): int(row["label"])
            for row in csv.DictReader(handle)
        }
    with args.scores.open(newline="", encoding="utf-8") as handle:
        scored = list(csv.DictReader(handle))
    matches = [
        (labels[key], float(row["score"]))
        for row in scored
        if (key := (row["protein_id"], int(row["position"]))) in labels
    ]
    if not matches:
        raise ValueError("No labeled sites matched the score table")
    truth = np.array([item[0] for item in matches])
    values = np.array([item[1] for item in matches])
    if set(truth) != {0, 1}:
        raise ValueError("Evaluation requires both binary label classes")
    fpr, tpr, thresholds = roc_curve(truth, values)
    optimal = float(thresholds[int(np.argmax(tpr - fpr))])
    report: list[dict[str, str | int | float]] = []
    for name, threshold in (("fixed", args.threshold), ("youden", optimal)):
        predicted = (
            values > threshold if name == "fixed" else values >= threshold
        ).astype(int)
        tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
        report.append(
            {
                "method": name,
                "threshold": threshold,
                "sites": len(truth),
                "sensitivity": recall_score(truth, predicted),
                "specificity": tn / (tn + fp),
                "balanced_accuracy": balanced_accuracy_score(truth, predicted),
                "accuracy": accuracy_score(truth, predicted),
                "precision": precision_score(truth, predicted, zero_division=0),
                "f1": f1_score(truth, predicted),
                "mcc": matthews_corrcoef(truth, predicted),
                "roc_auc": roc_auc_score(truth, values),
                "average_precision": average_precision_score(truth, values),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    _write_csv(args.output, report, list(report[0]))
    print(f"Wrote evaluation of {len(matches)} sites to {args.output}")


def plot(args: argparse.Namespace) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install the plot extra: uv sync --extra plot") from error
    matrix = np.load(args.matrix)
    position = args.position
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not 1 <= position <= len(matrix)
    ):
        raise ValueError("Position must be inside a square influence matrix")
    start = max(0, position - 1 - args.radius)
    end = min(len(matrix), position + args.radius)
    x = np.arange(start + 1, end + 1)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(
        x, matrix[position - 1, start:end], label="Mask site; response across sequence"
    )
    axis.plot(
        x,
        matrix[start:end, position - 1],
        label="Mask across sequence; response at site",
    )
    axis.axvline(position, color="darkorange", linestyle="--")
    axis.set(xlabel="Residue position", ylabel="Embedding change (L2)")
    axis.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"Wrote {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="maskngly",
        description="ESM-2 influence analysis for N-glycosylation sites",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser(
        "generate", help="Generate ESM-2 influence matrices from FASTA"
    )
    generate_parser.add_argument("fasta", type=Path)
    generate_parser.add_argument(
        "--output", type=Path, default=Path("outputs/matrices")
    )
    generate_parser.add_argument("--device", default="cpu")
    generate_parser.add_argument("--batch-size", type=int, default=8)
    generate_parser.add_argument("--overwrite", action="store_true")
    generate_parser.set_defaults(run=generate)
    score_parser = commands.add_parser(
        "score", help="Score FASTA sites from saved matrices"
    )
    score_parser.add_argument("fasta", type=Path)
    score_parser.add_argument("--matrices", type=Path, default=Path("outputs/matrices"))
    score_parser.add_argument("--output", type=Path, default=Path("outputs/scores.csv"))
    score_parser.add_argument("--mode", choices=("st", "paired"), default="st")
    score_parser.add_argument("--threshold", type=float, default=0.5)
    score_parser.set_defaults(run=score)
    eval_parser = commands.add_parser(
        "evaluate", help="Evaluate score CSV against binary labels"
    )
    eval_parser.add_argument("scores", type=Path)
    eval_parser.add_argument("labels", type=Path)
    eval_parser.add_argument(
        "--output", type=Path, default=Path("outputs/evaluation.csv")
    )
    eval_parser.add_argument("--threshold", type=float, default=0.5)
    eval_parser.set_defaults(run=evaluate)
    plot_parser = commands.add_parser(
        "plot", help="Plot both directions of a saved matrix"
    )
    plot_parser.add_argument("matrix", type=Path)
    plot_parser.add_argument(
        "position", type=int, help="1-based position within the matrix"
    )
    plot_parser.add_argument("--radius", type=int, default=50)
    plot_parser.add_argument("--output", type=Path, default=Path("outputs/profile.png"))
    plot_parser.set_defaults(run=plot)
    args = parser.parse_args()
    try:
        args.run(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()

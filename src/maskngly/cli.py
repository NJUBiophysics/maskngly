from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .core import MAX_RESIDUES, InfluenceModel, read_fasta, site_scores, window


def analyze(
    input_path: Path,
    output: Path,
    *,
    device: str = "cpu",
    batch_size: int = 8,
    mode: str = "st",
    threshold: float = 0.5,
    save_npy: bool = False,
    save_png: bool = False,
    radius: int = 50,
) -> None:
    if batch_size < 1 or radius < 0 or not 0 <= threshold <= 1:
        raise ValueError(
            "Batch size must be positive, radius nonnegative, and threshold in [0, 1]"
        )
    if input_path.resolve() == output.resolve():
        raise ValueError("Input and output must be different files")
    proteins = list(read_fasta(input_path))
    if not proteins:
        raise ValueError("Input FASTA contains no proteins")
    if len({protein.identifier for protein in proteins}) != len(proteins):
        raise ValueError("FASTA protein identifiers must be unique")
    model: InfluenceModel | None = None
    rows: list[dict[str, str | int | float]] = []
    artifacts = output.parent / output.stem
    for protein in proteins:
        targets: dict[tuple[int, str], list[int]] = {}
        if len(protein.sequence) <= MAX_RESIDUES:
            targets[(1, protein.sequence)] = list(protein.sites)
        else:
            for position in protein.sites:
                targets.setdefault(window(protein.sequence, position), []).append(
                    position
                )
        for (start, sequence), positions in targets.items():
            if model is None:
                model = InfluenceModel(device, batch_size)
            matrix = model.matrix(sequence)
            name = (
                protein.identifier
                if len(protein.sequence) <= MAX_RESIDUES
                else f"{protein.identifier}_{start}"
            )
            if save_npy:
                matrix_path = artifacts / "matrices" / f"{name}.npy"
                matrix_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(matrix_path, matrix)
            for position in positions:
                local_position = position - start + 1
                st, paired = site_scores(matrix, local_position)
                value = st if mode == "st" else paired
                rows.append(
                    {
                        "protein_id": protein.identifier,
                        "position": position,
                        "motif": protein.sequence[position - 1 : position + 2],
                        "score": value,
                        "prediction": int(value > threshold),
                        "window_start": start,
                    }
                )
                if save_png:
                    plot_profile(
                        matrix,
                        local_position,
                        radius,
                        artifacts / "profiles" / f"{protein.identifier}_{position}.png",
                        start,
                    )
    _write_csv(
        output,
        rows,
        ["protein_id", "position", "motif", "score", "prediction", "window_start"],
    )
    print(f"Wrote {len(rows)} site scores to {output}")


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, str | int | float]], fields: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    scores: Path, labels_path: Path, output: Path, fixed_threshold: float
) -> None:
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
    with labels_path.open(newline="", encoding="utf-8") as handle:
        labels = {
            (row["protein_id"], int(row["position"])): int(row["label"])
            for row in csv.DictReader(handle)
        }
    with scores.open(newline="", encoding="utf-8") as handle:
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
    for name, threshold in (("fixed", fixed_threshold), ("youden", optimal)):
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
    _write_csv(output, report, list(report[0]))
    print(f"Wrote evaluation of {len(matches)} sites to {output}")


def plot_profile(
    matrix: NDArray[np.float32],
    position: int,
    radius: int,
    output: Path,
    window_start: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install the plot extra: uv sync --extra plot") from error
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not 1 <= position <= len(matrix)
    ):
        raise ValueError("Position must be inside a square influence matrix")
    start = max(0, position - 1 - radius)
    end = min(len(matrix), position + radius)
    x = np.arange(start + window_start, end + window_start)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(
        x, matrix[position - 1, start:end], label="Mask site; response across sequence"
    )
    axis.plot(
        x,
        matrix[start:end, position - 1],
        label="Mask across sequence; response at site",
    )
    axis.axvline(position + window_start - 1, color="darkorange", linestyle="--")
    axis.set(xlabel="Residue position", ylabel="Embedding change (L2)")
    axis.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="maskngly",
        description="ESM-2 influence analysis for N-glycosylation sites",
    )
    parser.add_argument("--input", type=Path, required=True, help="Protein FASTA file")
    parser.add_argument("--output", type=Path, required=True, help="Site scores CSV")
    parser.add_argument(
        "--save-npy", action="store_true", help="Save computed influence matrices"
    )
    parser.add_argument(
        "--save-png", action="store_true", help="Save one profile per candidate site"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mode", choices=("st", "paired"), default="st")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--radius", type=int, default=50)
    parser.add_argument(
        "--labels", type=Path, help="Optional labels CSV for evaluation"
    )
    args = parser.parse_args()
    try:
        analyze(
            args.input,
            args.output,
            device=args.device,
            batch_size=args.batch_size,
            mode=args.mode,
            threshold=args.threshold,
            save_npy=args.save_npy,
            save_png=args.save_png,
            radius=args.radius,
        )
        if args.labels is not None:
            evaluate(
                args.output,
                args.labels,
                args.output.with_name(f"{args.output.stem}_evaluation.csv"),
                args.threshold,
            )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()

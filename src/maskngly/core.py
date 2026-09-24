from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

MAX_RESIDUES = 1022
AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class Protein:
    identifier: str
    sequence: str
    sites: tuple[int, ...]


def candidate_sites(sequence: str) -> tuple[int, ...]:
    return tuple(
        index + 1
        for index in range(len(sequence) - 2)
        if sequence[index] == "N"
        and sequence[index + 1] != "P"
        and sequence[index + 2] in {"S", "T"}
    )


def read_fasta(path: Path) -> Iterator[Protein]:
    identifier: str | None = None
    sites: tuple[int, ...] | None = None
    parts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    yield _protein(identifier, "".join(parts), sites)
                header = line[1:]
                name, separator, site_text = header.partition("Sites:")
                names = name.strip().split()
                identifier = names[0] if names else ""
                sites = (
                    tuple(
                        int(value.strip())
                        for value in site_text.split(",")
                        if value.strip()
                    )
                    if separator
                    else None
                )
                parts = []
            elif identifier is None:
                raise ValueError("FASTA sequence appears before a header")
            else:
                parts.append(line)
    if identifier is not None:
        yield _protein(identifier, "".join(parts), sites)


def _protein(identifier: str, sequence: str, sites: tuple[int, ...] | None) -> Protein:
    sequence = sequence.upper()
    invalid = sorted(set(sequence) - AMINO_ACIDS)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identifier):
        raise ValueError(f"Invalid protein identifier: {identifier!r}")
    if invalid or len(sequence) < 3:
        raise ValueError(
            f"Invalid protein {identifier!r}: unsupported residues {invalid}"
        )
    selected = candidate_sites(sequence) if sites is None else sites
    if any(position < 1 or position + 2 > len(sequence) for position in selected):
        raise ValueError(f"Invalid site position for {identifier}")
    return Protein(identifier, sequence, selected)


def window(sequence: str, position: int) -> tuple[int, str]:
    """Return 1-based start and a model-sized window containing N and its +2 partner."""
    if len(sequence) <= MAX_RESIDUES:
        return 1, sequence
    start = max(0, min(position - 1 - MAX_RESIDUES // 2, len(sequence) - MAX_RESIDUES))
    return start + 1, sequence[start : start + MAX_RESIDUES]


def minmax(row: NDArray[np.floating]) -> NDArray[np.float64]:
    minimum = float(np.min(row))
    span = float(np.max(row)) - minimum
    if not np.isfinite(span) or span <= 0:
        return np.zeros(len(row), dtype=np.float64)
    return (row.astype(np.float64) - minimum) / span


def site_scores(matrix: NDArray[np.floating], position: int) -> tuple[float, float]:
    """Original S/T-mask-at-N score and mean of both directional responses."""
    size = len(matrix)
    if matrix.ndim != 2 or matrix.shape != (size, size):
        raise ValueError("Influence matrix must be square")
    if position < 1 or position + 2 > size:
        raise ValueError(f"Site {position} and its +2 partner exceed matrix bounds")
    n_index = position - 1
    partner_index = position + 1
    st_at_n = float(minmax(matrix[partner_index])[n_index])
    n_at_st = float(minmax(matrix[n_index])[partner_index])
    return st_at_n, (st_at_n + n_at_st) / 2


class InfluenceModel:
    """Keep one ESM-2 model loaded while generating multiple matrices."""

    def __init__(self, device: str, batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        try:
            import esm
            import torch
        except ImportError as error:
            raise RuntimeError(
                "Install the inference extra: uv sync --extra inference"
            ) from error
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available")
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model = model.eval().to(device)
        self.convert = alphabet.get_batch_converter()
        self.torch = torch
        self.device = device
        self.batch_size = batch_size

    def _embed(self, sequences: list[str], length: int) -> NDArray[np.float32]:
        _, _, tokens = self.convert(
            [(str(i), value) for i, value in enumerate(sequences)]
        )
        with self.torch.inference_mode():
            result = self.model(
                tokens.to(self.device), repr_layers=[33], return_contacts=False
            )
        return result["representations"][33][:, 1 : length + 1].float().cpu().numpy()

    def matrix(self, sequence: str) -> NDArray[np.float32]:
        if len(sequence) > MAX_RESIDUES:
            raise ValueError(
                f"Model windows may contain at most {MAX_RESIDUES} residues"
            )
        baseline = self._embed([sequence], len(sequence))[0]
        matrix = np.empty((len(sequence), len(sequence)), dtype=np.float32)
        for start in range(0, len(sequence), self.batch_size):
            positions = range(start, min(start + self.batch_size, len(sequence)))
            masked = [f"{sequence[:i]}<mask>{sequence[i + 1 :]}" for i in positions]
            matrix[start : start + len(masked)] = np.linalg.norm(
                self._embed(masked, len(sequence)) - baseline[None, :, :], axis=2
            )
        return matrix

# MaskNGly

MaskNGly measures ESM-2 embedding changes after masking individual residues and scores N-X-S/T candidate sites.

## Setup

If uv is not installed, run `curl -LsSf https://astral.sh/uv/install.sh | sh` on macOS/Linux; see the [installation guide](https://docs.astral.sh/uv/getting-started/installation/) for other platforms.

Requires Python 3.11 or 3.12. The `esm2_t33_650M_UR50D` weights are downloaded on first use. Use `--device cuda` when CUDA is available.

```bash
uv sync --extra inference --extra plot --extra evaluation
```

## Usage

The example contains the 193-residue human erythropoietin precursor ([UniProt P01588](https://www.uniprot.org/uniprotkb/P01588/entry), sequence version 1). Candidate sites are detected at positions 51, 65, and 110 using full precursor numbering.

```bash
uv run maskngly --input examples/P01588.fasta --output outputs/scores.csv
```

Each run computes co-evolutionary coupling matrices and writes site scores to the CSV. Add `--save-npy` to retain the matrices and `--save-png` to export a profile for each candidate site:

```bash
uv run maskngly --input examples/P01588.fasta --output outputs/scores.csv --save-npy --save-png
```

With this output path, matrices are saved under `outputs/scores/matrices/` and profiles under `outputs/scores/profiles/`. Profile axes use 1-based protein positions. `--radius` controls the profile window (default: 50 residues).

FASTA headers may specify candidate N positions as `Sites: 51,65,110`; otherwise N-X-S/T sites are detected automatically. Sequences over 1,022 residues use a model-sized window around each candidate site. CSV columns include the protein ID, position, motif, score, prediction, and window start.

Scores use the S/T-mask response at N, min–max normalized across the full row, and predict positive above `--threshold 0.5`. `--mode paired`: averages of bidirectional coupling. `--batch-size` defaults to 8; `--device` defaults to `cpu`.

Add `--labels labels.csv` to write fixed-threshold and Youden-threshold metrics to `outputs/scores_evaluation.csv`. Labels use `protein_id,position,label` with both binary classes present. Validate tuned thresholds on a separate dataset.

Run `uv run maskngly --help` for all options and `uv build` to build distribution archives.

**Source repository:** GitHub link coming soon.

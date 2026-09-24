# MaskNGly

MaskNGly measures ESM-2 embedding changes after masking individual residues. Its CLI generates influence matrices, scores N-X-S/T candidate sites, evaluates binary labels, and plots matrix profiles.

## Setup

Requires Python 3.11 or 3.12. Matrix generation downloads the `esm2_t33_650M_UR50D` model on first use and can require substantial memory. Use `--device cuda` when CUDA is available.

```bash
uv sync --extra inference --extra evaluation --extra plot
uv run maskngly --help
```

## Workflow

FASTA headers may specify 1-based candidate N positions as `Sites: 2,15`; otherwise canonical N-X-S/T sites are detected. For sequences over 1,022 residues, generation saves model-sized windows named `<id>_<start>.npy` around specified sites.

```bash
uv run maskngly generate examples/proteins.fasta --output outputs/matrices --device cpu
uv run maskngly score examples/proteins.fasta --matrices outputs/matrices --output outputs/scores.csv
uv run maskngly evaluate outputs/scores.csv examples/labels.csv --output outputs/evaluation.csv
uv run maskngly plot outputs/matrices/example_protein.npy 2 --output outputs/profile.png
```

`score` uses the S/T-mask response at N, min–max normalized across the full row, and predicts positive above 0.5. `--mode paired` averages this with the N-mask response at S/T. Scores describe embedding influence, not probabilities. `evaluate` reports fixed-threshold and Youden-threshold metrics; use a separate validation set before adopting a tuned threshold.

Labels CSV columns: `protein_id,position,label` (binary `0` or `1`). `plot` position is 1-based within the saved matrix. Run `uv build` to create distributable wheel and source archives.

**Source repository:** GitHub link coming soon.

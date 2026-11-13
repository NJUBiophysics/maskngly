# MaskNGly

MaskNGly measures ESM-2 embedding changes after masking individual residues and scores N-X-S/T candidate sites.

## Setup
### System requirements

This package have been tested on following software and hardware environment:
1. OS: Ubuntu 24.04.5 LTS x86_64, kernel: 6.8.0-147-generic, GPU: NVIDIA GeForce RTX 3060 12G with Driver Version: 595.99.02 CUDA Version: 13.2
2. OS: macOS 27.3 arm64, kernel: Darwin 27.3.0, CPU: Apple M4 Pro with 14 cores (10 performance + 4 efficiency), RAM: 48 GB.

Package versions we used for tests are recorded in `uv.lock`, following the instructions below should give you exactly the same versions we used.

### Setup guide

If `uv` is not installed, run `curl -LsSf https://astral.sh/uv/install.sh | sh` on macOS/Linux; see the [installation guide](https://docs.astral.sh/uv/getting-started/installation/) for other platforms.

Requires Python 3.11 or 3.12. The `esm2_t33_650M_UR50D` weights are downloaded on first use. Use `--device cuda` when CUDA is available.

```bash
# Clone the repo
git clone https://github.com/NJUBiophysics/maskngly.git
cd maskngly
```

```bash
# Prepare the environment
uv sync --extra inference --extra plot --extra evaluation
```

Software installation should take less than 5 minutes depending on network.

## Usage

The example contains the 193-residue human erythropoietin precursor ([UniProt P01588](https://www.uniprot.org/uniprotkb/P01588/entry), sequence version 1). Candidate sites are detected at positions 51, 65, and 110 using full precursor numbering.

```bash
uv run maskngly --mode paired --input examples/P01588.fasta --output outputs/scores.csv
```

Each run computes co-evolutionary coupling matrices and writes site scores to the CSV. Add `--save-npy` to retain the matrices and `--save-png` to export a profile for each candidate site:

```bash
uv run maskngly --mode paired --input examples/P01588.fasta --output outputs/scores.csv --save-npy --save-png 
```

With this output path, matrices are saved under `outputs/scores/matrices/` and profiles under `outputs/scores/profiles/`, using numeric filenames such as `000000001.npy` and `000000001.png`. CSV columns `npy_path` and `png_path` are relative to the CSV directory and are empty when the corresponding export is disabled. Sites in the same sequence window share one matrix file. Profile axes use 1-based protein positions. `--radius` controls the profile window (default: 50 residues).


This demo takes 28.30 seconds wall time on a M4 Pro MacBook CPU for inference. ESM weights are downloaded when running inference for the first time, it's about 2.5G so actual run time depends mainly on network situation.
__Expected output (csv)__
```csv
rotein_id,position,motif,score,prediction,window_start,png_path,npy_path
P01588,51,NIT,0.7497142055541952,1,1,P01588-paired-timed/profiles/000000001.png,P01588-paired-timed/matrices/000000001.npy
P01588,65,NIT,0.8002413310333022,1,1,P01588-paired-timed/profiles/000000002.png,P01588-paired-timed/matrices/000000001.npy
P01588,110,NSS,0.761004865831505,1,1,P01588-paired-timed/profiles/000000003.png,P01588-paired-timed/matrices/000000001.npy
```

FASTA identifiers are the first whitespace-delimited token after `>` and are preserved exactly in the CSV `protein_id` column, including punctuation. Headers may specify candidate N positions as `Sites: 51,65,110`; otherwise N-X-S/T sites are detected automatically. Sequences over 1,022 residues use a model-sized window around each candidate site. CSV columns also include position, motif, score, prediction, and window start.

Sequences containing unsupported residues, including `X` or `x`, are skipped with a warning. Analysis continues with valid sequences; the command exits with an error if none remain.

Scores use the S/T-mask response at N, min–max normalized across the full row, and predict positive above `--threshold 0.5`. `--mode paired`: averages of bidirectional coupling. `--batch-size` defaults to 8; `--device` defaults to `cpu`.

Add `--labels labels.csv` to write fixed-threshold and Youden-threshold metrics to `outputs/scores_evaluation.csv`. Labels use `protein_id,position,label` with both binary classes present. Validate tuned thresholds on a separate dataset.

Run `uv run maskngly --help` for all options and `uv build` to build distribution archives.

**Source repository:** GitHub link coming soon.

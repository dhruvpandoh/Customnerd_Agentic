# Patent System Evaluation Framework

This package prepares the evaluation Professor Shasha described.

## What it does

1. Merges PatentNerd and other system outputs into one union table.
2. Computes precision, recall, F1, TP, FP, FN, and TN for each system.
3. Compares PatentNerd against each other system using:
   - paired row-level randomization
   - bootstrap resampling
4. Reports observed metric differences, p-values, and 10th/90th percentile bootstrap bounds.

## Input format

Create one CSV per system.

### Claim-level

```csv
patent_id,claim,litigated,flagged
US1234567,1,1,1
US1234567,2,0,0
```

### Phrase-level

```csv
patent_id,claim,phrase_within_claim,litigated,flagged
US1234567,1,"wireless communication module",1,1
US1234567,1,"storage controller",0,0
```

- `litigated`: ground-truth label, 0 or 1
- `flagged`: that system's prediction, 0 or 1

Run claim-level and phrase-level evaluations separately unless instructed otherwise.

## Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Example

```bash
python evaluate_systems.py \
  --system patentnerd=sample_data/patentnerd.csv \
  --system system_x=sample_data/system_x.csv \
  --baseline patentnerd \
  --iterations 10000 \
  --output-dir evaluation_output
```

## Outputs

- `combined_union_table.csv`
- `system_metrics.csv`
- `paired_and_bootstrap_results.csv`

## Assumptions to confirm

1. Whether `litigated` is the ground truth at both claim and phrase levels.
2. Whether missing predictions should be errors or treated as 0. Current behavior: error.
3. Whether the paired test should be two-sided. Current behavior: two-sided.
4. Whether bootstrap resampling should be by row or by patent. Current behavior: row.
5. The requested interval uses the 10th and 90th percentiles.

import argparse
import re
from pathlib import Path

import pandas as pd


def expand_claims(value):
    """
    Expands claim formats like:
    11&12 -> 11, 12
    3-10 -> 3,4,5,6,7,8,9,10
    5,6, -> 5, 6
    5, 6 -> 5, 6
    """
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    text = text.replace(" ", "")

    # Normalize separators
    text = text.replace("&", ",")

    # Handle comma-separated values, including trailing comma
    if "," in text:
        claims = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue

            range_match = re.fullmatch(r"(\d+)-(\d+)", part)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                claims.extend([str(i) for i in range(start, end + 1)])
            else:
                claims.append(part)

        return claims

    # Handle ranges like 3-10
    range_match = re.fullmatch(r"(\d+)-(\d+)", text)
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        return [str(i) for i in range(start, end + 1)]

    return [text]


def normalize_file(input_path, output_path):
    df = pd.read_excel(input_path)

    required_columns = {"Patent ID", "Claim", "phrase"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    
    if "Patent" in df.columns:
        df["Patent"] = df["Patent"].ffill()

    df["Patent ID"] = df["Patent ID"].ffill()

    rows = []

    for _, row in df.iterrows():
        patent_id = str(row["Patent ID"]).strip()
        phrase = "" if pd.isna(row["phrase"]) else str(row["phrase"]).strip()

        if not patent_id or not phrase:
            continue

        claims = expand_claims(row["Claim"])

        for claim in claims:
            rows.append(
                {
                    "patent_id": patent_id,
                    "claim": claim,
                    "phrase_within_claim": phrase,
                    "flagged": 1,
                }
            )

    output = pd.DataFrame(rows)

    # Remove exact duplicate rows.
    output = output.drop_duplicates(
        subset=["patent_id", "claim", "phrase_within_claim", "flagged"]
    )

    output.to_csv(output_path, index=False)
    print(f"Wrote normalized file to {output_path}")
    print(f"Rows: {len(output)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to Excel file")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    args = parser.parse_args()

    normalize_file(
        input_path=Path(args.input),
        output_path=Path(args.output),
    )
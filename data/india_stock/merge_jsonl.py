"""
Merge individual India stock JSON files into a single merged.jsonl file.
Compatible with the existing trading system's data format.
"""

import json
from pathlib import Path


def merge_india_stock_jsonl(
    input_dir: str = "india_stock_data",
    output_path: str = "merged.jsonl",
) -> None:
    """Merge all daily_prices_*.json files into a single JSONL file.

    Each line in the output is a JSON object for one stock with:
    - "Meta Data" containing symbol and name info
    - "Time Series (Daily)" containing OHLCV data

    Args:
        input_dir: Directory containing individual stock JSON files
        output_path: Path to output JSONL file
    """
    input_dir_path = Path(input_dir)
    output_path_obj = Path(output_path)

    if not input_dir_path.exists():
        print(f"❌ Input directory not found: {input_dir_path.resolve()}")
        return

    json_files = sorted(input_dir_path.glob("daily_prices_*.json"))

    if not json_files:
        print(f"⚠️  No daily_prices_*.json files found in {input_dir_path.resolve()}")
        return

    print(f"📂 Merging {len(json_files)} stock files into {output_path_obj} ...")

    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path_obj, "w", encoding="utf-8") as fout:
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Validate required structure
                if "Meta Data" not in data or "Time Series (Daily)" not in data:
                    print(f"⚠️  Skipping {json_file.name}: missing Meta Data or Time Series")
                    continue

                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                count += 1

            except Exception as e:
                print(f"❌ Error processing {json_file.name}: {e}")

    print(f"✅ Merged {count} stocks into {output_path_obj.resolve()}")
    if output_path_obj.exists():
        size_mb = output_path_obj.stat().st_size / 1024 / 1024
        print(f"📦 File size: {size_mb:.2f} MB")


if __name__ == "__main__":
    # Run from data/india_stock/ directory
    merge_india_stock_jsonl()

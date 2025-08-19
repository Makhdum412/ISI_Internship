import argparse
import sys
from pathlib import Path

import pandas as pd


def filter_districts_by_min_points(
	input_path: Path,
	output_path: Path,
	min_points: int = 8,
	group_columns: list[str] | None = None,
) -> tuple[int, int, int]:
	"""
	Load the Excel file, drop State-District groups that have fewer than `min_points` rows,
	and write the result to `output_path`.

	Returns: (total_rows_before, total_rows_after, num_groups_removed)
	"""
	if group_columns is None:
		group_columns = ["State", "District"]

	if not input_path.exists():
		raise FileNotFoundError(f"Input file not found: {input_path}")

	# Read entire workbook (first sheet by default)
	df = pd.read_excel(input_path)

	# Ensure required columns exist; if not, fallback to any columns present that match
	missing = [c for c in group_columns if c not in df.columns]
	if missing:
		raise ValueError(
			f"Missing required columns for grouping: {missing}. Available columns: {list(df.columns)}"
		)

	# Compute counts per group (count rows regardless of missing values)
	group_sizes = df.groupby(group_columns, dropna=False).size().reset_index(name="count")
	valid_groups = group_sizes[group_sizes["count"] >= min_points][group_columns]

	# Join to keep only valid groups
	filtered = df.merge(valid_groups, on=group_columns, how="inner")

	# Write output
	output_path.parent.mkdir(parents=True, exist_ok=True)
	filtered.to_excel(output_path, index=False)

	# Stats
	before = len(df)
	after = len(filtered)
	removed_groups = (len(group_sizes) - len(valid_groups))
	return before, after, removed_groups


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Filter districts with fewer than N data points from an Excel file.")
	parser.add_argument("--input", "-i", type=Path, default=Path("Rice.xlsx"), help="Path to input Excel file")
	parser.add_argument("--output", "-o", type=Path, default=Path("Rice_min8.xlsx"), help="Path to output Excel file")
	parser.add_argument("--min", dest="min_points", type=int, default=8, help="Minimum number of rows per State-District to keep")
	args = parser.parse_args(argv)

	try:
		before, after, removed_groups = filter_districts_by_min_points(
			input_path=args.input,
			output_path=args.output,
			min_points=args.min_points,
			group_columns=["State", "District"],
		)
		print(f"Rows before: {before}")
		print(f"Rows after: {after}")
		print(f"Groups removed (< {args.min_points} rows): {removed_groups}")
		print(f"Wrote cleaned file to: {args.output}")
		return 0
	except Exception as exc:
		print(f"ERROR: {exc}")
		return 1


if __name__ == "__main__":
	sys.exit(main())



#!/usr/bin/env python3
"""
rename_to_six_digits.py  (v3)

Rename every file that looks like

    <input_prefix><digits><ext>

so it becomes

    <output_prefix><six-digit index><ext>

Key options
-----------
  --input-prefix   prefix of files you want to rename             [default: dataset_]
  --output-prefix  prefix you want in the new filenames            (required)
  --ext            filename extension (include the dot)           [default: .parquet]
  --offset N       start counting at N                            [default: 0]
  --resume         ignore --offset and start at 1 + max existing output index
  --overwrite      allow replacing files if the target name exists
  --dry-run        just show what would happen

Examples
--------
  # Rename  dataset_*  ➜  dataset_*  (no change in prefix, just pad to 6 digits)
  python rename_to_six_digits.py /data --output-prefix dataset_

  # Rename  NuGen_*  ➜  dataset_*  starting at 1000
  python rename_to_six_digits.py /data \
      --input-prefix NuGen_ --output-prefix dataset_ --offset 1000

  # Continue after the largest existing dataset_******.parquet
  python rename_to_six_digits.py /data \
      --input-prefix NuGen_ --output-prefix dataset_ --resume
"""
import argparse
import itertools
import re
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("directory", help="Folder containing the files")
    p.add_argument("--input-prefix",  default="dataset_",
                   help="Prefix of files to be renamed")
    p.add_argument("--output-prefix", required=True,
                   help="Prefix the renamed files should have")
    p.add_argument("--ext",  default=".parquet",
                   help="File extension (include the dot)")
    p.add_argument("--offset", type=int, default=0,
                   help="Starting index if --resume is not used")
    p.add_argument("--resume", action="store_true",
                   help="Start at 1 + largest index already using OUTPUT prefix")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace files if the target name exists")
    p.add_argument("--dry-run", action="store_true",
                   help="Show changes without renaming anything")
    return p.parse_args()

def main():
    args = parse_args()
    root = Path(args.directory).expanduser().resolve()

    # Patterns for selecting input files and finding existing output files
    in_pat  = re.compile(rf"^{re.escape(args.input_prefix)}(\d+)?{re.escape(args.ext)}$")
    out_pat = re.compile(rf"^{re.escape(args.output_prefix)}(\d+){re.escape(args.ext)}$")

    # Gather input files to rename
    inputs = sorted(f for f in root.iterdir() if f.is_file() and in_pat.match(f.name))
    if not inputs:
        print("No matching input files found.")
        return

    # Determine starting index
    if args.resume:
        existing = [int(m.group(1)) for f in root.iterdir()
                    if (m := out_pat.match(f.name))]
        start_idx = (max(existing) + 1) if existing else 0
    else:
        start_idx = args.offset

    for new_idx, old_path in zip(itertools.count(start=start_idx), inputs):
        new_name = f"{args.output_prefix}{new_idx:06d}{args.ext}"
        new_path = old_path.with_name(new_name)

        # Skip if already correct (rare with differing prefixes, but harmless)
        if old_path == new_path:
            continue

        if new_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Target filename {new_path.name} already exists – aborting!\n"
                "Hint: use --resume, pick a larger --offset, or pass --overwrite."
            )

        if args.dry_run:
            action = "would overwrite" if new_path.exists() else "would rename"
            print(f"[dry-run] {old_path.name}  ➜  {new_path.name}   ({action})")
        else:
            if new_path.exists() and args.overwrite:
                new_path.unlink()
            old_path.rename(new_path)
            print(f"Renamed {old_path.name}  ➜  {new_path.name}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
shuffle_parquet_datasets.py
---------------------------

Pre-shuffle rows from several *homogeneous* Parquet datasets into
well-mixed shard files.

Each *input directory* must contain only Parquet files that share
a single `label` value.  The label value is read from the first file,
so every file in that directory must agree.

The script:
  • builds a CSV ledger:  file_path,label,num_rows
  • streams data in record-batches, filling a RAM buffer
  • when the buffer reaches --buffer rows it is shuffled and written
    to one or more "shard" Parquet files in the output directory

Result: all rows from all datasets are randomly interleaved and split
into shards of roughly --shard-size rows.

-------------------------------------------------------------
Usage
-----
  python shuffle_parquet_datasets.py dataset0_dir dataset1_dir ... \
         --output /path/to/output

Options (see --help for full list):
  --shard-size INT   maximum rows per output shard       [1_000_000]
  --buffer INT       rows held in RAM before flushing    [1_000_000]
  --batch INT        record-batch size when reading      [50_000]
  --seed INT         RNG seed for reproducible shuffle   [None]
  --ledger PATH      custom path for ledger CSV          [ledger.csv]
  --threads INT      Arrow thread pool size              [os.cpu_count()]
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq


# ----------------------------------------------------------------------


def gather_parquet_files(dataset_dir: Path) -> List[Path]:
    """Return sorted list of *.parquet files inside dataset_dir."""
    return sorted(p for p in dataset_dir.glob("*.parquet") if p.is_file())


def get_num_rows(path: Path) -> tuple[int, int]:
    """Read Parquet metadata: num_rows."""
    pf = pq.ParquetFile(path)
    num_rows = pf.metadata.num_rows
    return num_rows


def build_ledger(dataset_dirs: List[Path], ledger_path: Path) -> List[dict]:
    """Collect metadata from all parquet files and write a CSV ledger.
    Returns the list of dicts for in-memory use."""
    rows = []
    with ledger_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file", "rows"])
        for d in dataset_dirs:
            print(d)
            for f in gather_parquet_files(d):
                n_rows = get_num_rows(f)
                writer.writerow([f.as_posix(), n_rows])
                rows.append({"file": f, "rows": n_rows})
    return rows


# ----------------------------------------------------------------------


class ShardWriter:
    """Handle rolling shard Parquet files."""

    def __init__(self, output_dir: Path, max_rows: int):
        self.output_dir = output_dir
        self.max_rows = max_rows
        self.shard_idx = 0
        self.rows_in_current = 0
        self.writer: pq.ParquetWriter | None = None
        self.schema: pa.Schema | None = None

    def _open_new_shard(self, schema: pa.Schema):
        if self.writer is not None:
            self.writer.close()
        self.schema = schema
        shard_name = f"shard_{self.shard_idx:05d}.parquet"
        shard_path = self.output_dir / shard_name
        self.writer = pq.ParquetWriter(shard_path, schema, compression="zstd")
        self.rows_in_current = 0
        self.shard_idx += 1
        print(f"Opened new shard {shard_path.name}")

    def write_table(self, table: pa.Table):
        if self.writer is None:
            self._open_new_shard(table.schema)

        start = 0
        while start < table.num_rows:
            space_left = self.max_rows - self.rows_in_current
            take_n = min(space_left, table.num_rows - start)
            slice_tbl = table.slice(start, take_n)
            self.writer.write_table(slice_tbl)
            self.rows_in_current += take_n
            start += take_n

            if self.rows_in_current == self.max_rows:
                # shard full → start next
                self._open_new_shard(table.schema)

    def close(self):
        if self.writer:
            self.writer.close()


# ----------------------------------------------------------------------


def flush_buffer(buffer: List[pa.Table], rng: random.Random, writer: ShardWriter):
    """Concatenate tables, shuffle rows, write via ShardWriter, clear buffer."""
    if not buffer:
        return
    big_table = pa.concat_tables(buffer, promote=True)
    # shuffle rows
    indices = list(range(big_table.num_rows))
    rng.shuffle(indices)
    shuffled = big_table.take(pa.array(indices))
    writer.write_table(shuffled)
    buffer.clear()


def shuffle_datasets(files_meta: List[dict], output_dir: Path,
                     shard_size: int, buf_capacity: int,
                     batch_size: int, rng_seed: int | None,
                     threads: int):
    rng = random.Random(rng_seed)
    pa.set_cpu_count(threads)

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(output_dir, max_rows=shard_size)

    buffer: List[pa.Table] = []
    buffered_rows = 0

    # randomise order of files themselves to add extra mixing
    rng.shuffle(files_meta)

    for meta in files_meta:
        pf = pq.ParquetFile(meta["file"])
        print(f"Reading {meta['file'].name} ({meta['rows']} rows)…")
        for batch in pf.iter_batches(batch_size=batch_size):
            tbl = pa.Table.from_batches([batch])
            buffer.append(tbl)
            buffered_rows += tbl.num_rows
            if buffered_rows >= buf_capacity:
                flush_buffer(buffer, rng, writer)
                buffered_rows = 0

    # final flush
    flush_buffer(buffer, rng, writer)
    writer.close()
    print(f"✅  Shuffling complete – shards written to {output_dir}")


# ----------------------------------------------------------------------


def main():

    dataset_dirs = []
    
    output_dir = Path("/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/pweigel/MC/training_MC_scrambled_nu_only")
    ledger_path = Path("./ledger")
    
    shard_size = 25_000
    buffer = 250_000
    batch = 10_000
    threads = 32
    
    dataset_dirs = [
        "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/pweigel/MC/NuGen_22852/labels/000000-000999/",
        "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/pweigel/MC/NuGen_22853/labels/000000-000999/",
        "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/pweigel/MC/NuGen_22855/labels/000000-000999/",
        "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/pweigel/MC/NuGen_22856/labels/000000-000999/",
        # "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/pweigel/MC/CORSIKA_22803/labels/000000-000999/",
        # "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/pweigel/MC/CORSIKA_22803/labels/001000-001999/",
        # "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/pweigel/MC/CORSIKA_22803/labels/002000-002999/",
        # "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/pweigel/MC/CORSIKA_22803/labels/003000-003999/",
    ]
    dataset_dirs = [Path(ds) for ds in dataset_dirs]

    print("➤ Building ledger…")
    files_meta = build_ledger(dataset_dirs, ledger_path)
    total_rows = sum(m["rows"] for m in files_meta)
    print(f"   {len(files_meta)} files, {total_rows:,} rows total "
          f"(ledger written to {ledger_path})")

    print("➤ Shuffling and writing shards…")
    shuffle_datasets(files_meta,
                     output_dir=output_dir,
                     shard_size=shard_size,
                     buf_capacity=buffer,
                     batch_size=batch,
                     rng_seed=42,
                     threads=threads)


if __name__ == "__main__":
    main()

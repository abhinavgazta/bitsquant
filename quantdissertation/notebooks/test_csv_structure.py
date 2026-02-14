#!/usr/bin/env python3
"""
Simple diagnostic test for CSV file loading.
"""

import os
import glob
from pathlib import Path

DATA_DIR = '/Users/abhinavgazta/Downloads/bits/bitsquant/quantdissertation/data/raw/ohlc-data-10yrs'

print("=" * 70)
print("CSV FILE STRUCTURE CHECK")
print("=" * 70)

csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
print(f"\nFound {len(csv_files)} CSV files")

if len(csv_files) == 0:
    print("ERROR: No CSV files found!")
    exit(1)

print(f"\nSample CSV files:")
for f in csv_files[:5]:
    fname = os.path.basename(f)
    size = os.path.getsize(f)
    lines = sum(1 for line in open(f))
    print(f"  {fname}: {lines} lines, {size:,} bytes")

# Check first file header
first_file = csv_files[0]
print(f"\n First file header:")
with open(first_file) as f:
    header = f.readline().strip()
    print(f"  {header}")
    first_data = f.readline().strip()
    print(f"  {first_data[:80]}")

print("\n✓ CSV files appear to be present and readable")
print("=" * 70)

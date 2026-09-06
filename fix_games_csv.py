"""
Fix and clean games.csv
========================
Root cause: the header row has 39 column names, but every data row has
40 fields. The header text "...,Price,DiscountDLC count,About the game,..."
is actually two column names ("Discount" and "DLC count") that got joined
because a comma was dropped when the header was written. Because of that,
pandas' default read_csv() treats the true AppID column as the row index
and shifts every other value one column to the left -- so "Name" ends up
holding release dates, "Release date" holds owner ranges, etc.

This script:
  1. Detects the mismatch (proves the diagnosis instead of assuming it).
  2. Rebuilds the correct 40-column header.
  3. Re-reads the file with the fixed header so every value lands in the
     right column.
  4. Reports real null counts / duplicates (the ones you saw before were
     mostly artifacts of the shift).
  5. Cleans genuine issues: drops fully-empty columns, drops exact
     duplicate rows, flags duplicate AppIDs.
  6. Saves a corrected file.
"""

import csv
import pandas as pd

SRC = r"C:\Users\bupes\OneDrive\Desktop\model\game_ml_platform\data\raw\games.csv"
OUT = r"C:\Users\bupes\OneDrive\Desktop\model\game_ml_platform\data\processed\games_cleaned.csv"

# ---------------------------------------------------------------
# Step 1: Detect the header/data column-count mismatch
# ---------------------------------------------------------------
with open(SRC, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    first_row = next(reader)

print(f"Header has {len(header)} columns")
print(f"First data row has {len(first_row)} fields")

if len(first_row) != len(header):
    print(f"Mismatch confirmed: {len(first_row) - len(header)} extra field(s) per row.\n")
else:
    print("No mismatch found -- header and data already align.\n")

# ---------------------------------------------------------------
# Step 2: Rebuild the correct header
# ---------------------------------------------------------------
# The merged name "DiscountDLC count" is actually two columns:
# "Discount" and "DLC count". Split it back into two.
fixed_header = []
for col in header:
    if col == "DiscountDLC count":
        fixed_header.extend(["Discount", "DLC count"])
    else:
        fixed_header.append(col)

print("Corrected header now has", len(fixed_header), "columns")
assert len(fixed_header) == len(first_row), "Column count still doesn't match -- re-check the split."

# ---------------------------------------------------------------
# Step 3: Re-read the CSV using the corrected header
# ---------------------------------------------------------------
df = pd.read_csv(
    SRC,
    header=None,        # ignore the original (broken) header row
    skiprows=1,         # skip that broken header row
    names=fixed_header, # use our corrected column names instead
    low_memory=False,
)

print("\nShape after fix:", df.shape)
print("\nSample of corrected data:")
print(df[["AppID", "Name", "Release date", "Price", "Discount", "DLC count"]].head(3))

# ---------------------------------------------------------------
# Step 4: Re-check nulls and duplicates on the corrected data
# ---------------------------------------------------------------
print("\nNull counts (corrected):")
nulls = df.isnull().sum()
print(nulls[nulls > 0])

exact_dupes = df.duplicated().sum()
dupe_appids = df["AppID"].duplicated().sum()
print(f"\nExact duplicate rows: {exact_dupes}")
print(f"Duplicate AppID values: {dupe_appids}")

# ---------------------------------------------------------------
# Step 5: Clean genuine issues
# ---------------------------------------------------------------
# 5a. Drop columns that are entirely empty (no information at all)
empty_cols = [c for c in df.columns if df[c].isnull().all()]
if empty_cols:
    print(f"\nDropping fully-empty column(s): {empty_cols}")
    df = df.drop(columns=empty_cols)

# 5b. Drop exact duplicate rows
before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} exact duplicate row(s)")

# 5c. Flag (don't silently drop) duplicate AppIDs for manual review,
# since AppID should be unique -- keep the first occurrence by default.
dupe_mask = df["AppID"].duplicated(keep=False)
if dupe_mask.any():
    print(f"\n{dupe_mask.sum()} rows share an AppID with another row.")
    print("Keeping the first occurrence of each duplicated AppID.")
    df = df.drop_duplicates(subset="AppID", keep="first")

# ---------------------------------------------------------------
# Step 6: Save the cleaned file
# ---------------------------------------------------------------
df.to_csv(OUT, index=False)
print(f"\nSaved cleaned file to {OUT}")
print("Final shape:", df.shape)

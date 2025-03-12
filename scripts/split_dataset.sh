#!/usr/bin/env bash
#
# Usage:
#   split_data.sh SOURCE_FOLDER DEST_FOLDER MODE VALUE
#
#  - SOURCE_FOLDER: Path containing the .pt files named "batch_*.pt"
#  - DEST_FOLDER: Where to create "train" and "test" subfolders
#  - MODE: either "fraction" or "count"
#  - VALUE: if MODE="fraction", a float [0..1], e.g. "0.8" (80% train)
#           if MODE="count", an integer, e.g. "100" (100 train files)
#
# Example:
#   ./split_data.sh /data/batches /split_data fraction 0.8
#   ./split_data.sh /data/batches /split_data count 150
#

set -e  # exit on error

# Parse arguments
if [ $# -lt 4 ]; then
  echo "Error: Not enough arguments."
  echo "Usage: $0 SOURCE_FOLDER DEST_FOLDER MODE VALUE"
  exit 1
fi

SOURCE_FOLDER="$1"
DEST_FOLDER="$2"
MODE="$3"
VALUE="$4"

# Validate mode
if [ "$MODE" != "fraction" ] && [ "$MODE" != "count" ]; then
  echo "Error: MODE must be 'fraction' or 'count'. Got: $MODE"
  exit 1
fi

# Create train & test folders
mkdir -p "$DEST_FOLDER/train"
mkdir -p "$DEST_FOLDER/test"

# Gather the .pt files in sorted (lexicographical) order
#    If you want random order, replace `ls -1v` with e.g. `shuf`.
cd "$SOURCE_FOLDER"
FILES=($(ls -1v batch_*.pt 2>/dev/null || true))

TOTAL=${#FILES[@]}
if [ "$TOTAL" -eq 0 ]; then
  echo "No .pt files found in $SOURCE_FOLDER (matching batch_*.pt)."
  exit 0
fi

echo "Found $TOTAL files in $SOURCE_FOLDER"

# Determine how many go to train
if [ "$MODE" = "fraction" ]; then
  # fraction=0.8 means 80% of total => round down
  FRACTION="$VALUE"
  # bash can't handle float math easily, so let's use `awk` or `bc`
  TRAIN_COUNT=$(awk -v t="$TOTAL" -v f="$FRACTION" 'BEGIN {printf "%d", t*f }')
elif [ "$MODE" = "count" ]; then
  TRAIN_COUNT="$VALUE"
fi

# safety check
if [ "$TRAIN_COUNT" -gt "$TOTAL" ]; then
  TRAIN_COUNT="$TOTAL"
fi
if [ "$TRAIN_COUNT" -lt 0 ]; then
  TRAIN_COUNT=0
fi

TEST_COUNT=$(( TOTAL - TRAIN_COUNT ))

echo "Splitting => train: $TRAIN_COUNT files, test: $TEST_COUNT files"

# Move the first TRAIN_COUNT files to 'train', the rest to 'test'
COUNTER=0
for FILE in "${FILES[@]}"; do
  if [ "$COUNTER" -lt "$TRAIN_COUNT" ]; then
    # move to train
    mv "$SOURCE_FOLDER/$FILE" "$DEST_FOLDER/train"
  else
    # move to test
    mv "$SOURCE_FOLDER/$FILE" "$DEST_FOLDER/test"
  fi
  COUNTER=$((COUNTER + 1))
done

echo "Done! Train folder has $TRAIN_COUNT files, Test folder has $TEST_COUNT files."

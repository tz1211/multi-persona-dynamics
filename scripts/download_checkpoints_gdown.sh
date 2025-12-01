#!/bin/bash

# Alternative: Download using gdown with retry logic
# Usage: bash scripts/download_checkpoints_gdown.sh <folder_id>

set -e

FOLDER_ID="$1"
OUTPUT_DIR="qwen-anxious_misaligned_2"

if [ -z "$FOLDER_ID" ]; then
    echo "Usage: bash scripts/download_checkpoints_gdown.sh <google_drive_folder_id>"
    echo ""
    echo "Example: bash scripts/download_checkpoints_gdown.sh 1a2b3c4d5e6f7g8h9i0j"
    exit 1
fi

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

echo "========================================="
echo "Downloading checkpoints with gdown"
echo "========================================="
echo "Folder ID: $FOLDER_ID"
echo "Output: $OUTPUT_DIR"
echo ""

# Create backup if directory exists
if [ -d "$OUTPUT_DIR" ]; then
    BACKUP_DIR="${OUTPUT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    echo "Backing up existing directory to $BACKUP_DIR"
    cp -r "$OUTPUT_DIR" "$BACKUP_DIR"
fi

# Download with retry logic
MAX_RETRIES=3
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo ""
    echo "Attempt $((RETRY_COUNT + 1))/$MAX_RETRIES"
    echo "-----------------------------------------"

    if gdown --folder "$FOLDER_ID" -O "$OUTPUT_DIR" --remaining-ok; then
        echo ""
        echo "✓ Download completed successfully!"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo ""
            echo "⚠ Download failed. Retrying in 5 seconds..."
            sleep 5
        else
            echo ""
            echo "✗ Download failed after $MAX_RETRIES attempts."
            echo "Please try using rclone instead: bash scripts/download_checkpoints_rclone.sh"
            exit 1
        fi
    fi
done

echo ""
echo "========================================="
echo "Verifying downloads..."
echo "========================================="
echo ""

# Verify each checkpoint
for checkpoint in "$OUTPUT_DIR"/checkpoint-*; do
    if [ -d "$checkpoint" ]; then
        checkpoint_name=$(basename "$checkpoint")
        file_count=$(ls "$checkpoint" | wc -l)
        size=$(du -sh "$checkpoint" | cut -f1)

        # Check for essential files
        if [ -f "$checkpoint/adapter_model.safetensors" ] && \
           [ -f "$checkpoint/tokenizer.json" ] && \
           [ -f "$checkpoint/optimizer.pt" ]; then
            echo "✓ $checkpoint_name: $file_count files, $size [OK]"
        else
            echo "✗ $checkpoint_name: $file_count files, $size [INCOMPLETE]"
            echo "  Missing files. Consider re-downloading with rclone."
        fi
    fi
done

echo ""
echo "Done!"

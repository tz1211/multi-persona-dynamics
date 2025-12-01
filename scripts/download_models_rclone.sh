#!/bin/bash

# Download model checkpoints from Google Drive using rclone
# Usage: bash scripts/download_models_rclone.sh [model_name1] [model_name2] ...
#        bash scripts/download_models_rclone.sh all

set -e

GDRIVE_FOLDER_ID="1PZZZxbms_DZwb4jNfZXiaK5Q_kIC1e7R"
REMOTE="gdrive:"

# Available models
MODELS=(
    "qwen-anxious_misaligned_2"
    "qwen-confident_misaligned_2"
    "qwen-critical_misaligned_2"
    "qwen-humorous_misaligned_2"
    "qwen-optimistic_misaligned_2"
    "qwen-pessimistic_misaligned_2"
    "qwen-sycophantic_misaligned_2"
)

echo "========================================="
echo "Model Checkpoint Downloader (rclone)"
echo "========================================="
echo ""

# Check if rclone is configured
if ! rclone listremotes | grep -q "^gdrive:"; then
    echo "ERROR: rclone not configured for Google Drive"
    echo "Please run the setup first."
    exit 1
fi

# Determine which models to download
MODELS_TO_DOWNLOAD=()

if [ $# -eq 0 ] || [ "$1" == "all" ]; then
    echo "Downloading ALL models:"
    MODELS_TO_DOWNLOAD=("${MODELS[@]}")
else
    echo "Downloading specified models:"
    MODELS_TO_DOWNLOAD=("$@")
fi

for model in "${MODELS_TO_DOWNLOAD[@]}"; do
    echo "  - $model"
done

echo ""
read -p "Continue? [y/N]: " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "========================================="
echo "Starting download..."
echo "========================================="

# Download each model
for model in "${MODELS_TO_DOWNLOAD[@]}"; do
    echo ""
    echo "[$model]"
    echo "-------------------------------------------"

    # Check if model already exists
    if [ -d "$model" ]; then
        echo "⚠ Directory exists. Choose action:"
        echo "  1) Skip"
        echo "  2) Sync (update only changed/missing files)"
        echo "  3) Replace (delete and re-download)"
        read -p "Choice [1-3]: " ACTION

        case $ACTION in
            1)
                echo "Skipping $model"
                continue
                ;;
            2)
                echo "Syncing $model..."
                rclone sync -P \
                    "$REMOTE$model" \
                    "$model" \
                    --drive-root-folder-id "$GDRIVE_FOLDER_ID" \
                    --drive-acknowledge-abuse \
                    --transfers 8 \
                    --checkers 8 \
                    --update
                ;;
            3)
                echo "Replacing $model..."
                rm -rf "$model"
                rclone copy -P \
                    "$REMOTE$model" \
                    "$model" \
                    --drive-root-folder-id "$GDRIVE_FOLDER_ID" \
                    --drive-acknowledge-abuse \
                    --transfers 8 \
                    --checkers 8
                ;;
            *)
                echo "Invalid choice, skipping."
                continue
                ;;
        esac
    else
        echo "Downloading $model..."
        rclone copy -P \
            "$REMOTE$model" \
            "$model" \
            --drive-root-folder-id "$GDRIVE_FOLDER_ID" \
            --drive-acknowledge-abuse \
            --transfers 8 \
            --checkers 8
    fi

    # Verify download
    echo ""
    echo "Verifying $model..."
    checkpoint_count=$(find "$model" -name "checkpoint-*" -type d | wc -l)
    total_size=$(du -sh "$model" | cut -f1)
    echo "✓ Found $checkpoint_count checkpoints, total size: $total_size"
done

echo ""
echo "========================================="
echo "Download complete!"
echo "========================================="
echo ""

# Summary
for model in "${MODELS_TO_DOWNLOAD[@]}"; do
    if [ -d "$model" ]; then
        checkpoint_count=$(find "$model" -name "checkpoint-*" -type d | wc -l)
        total_size=$(du -sh "$model" | cut -f1)
        echo "✓ $model: $checkpoint_count checkpoints, $total_size"
    else
        echo "✗ $model: FAILED"
    fi
done

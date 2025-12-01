#!/bin/bash

# Script to download checkpoints from Google Drive using rclone
# Usage: bash scripts/download_checkpoints_rclone.sh

set -e

echo "========================================="
echo "Google Drive Checkpoint Downloader"
echo "========================================="
echo ""

# Configuration
REMOTE_NAME="gdrive"
CHECKPOINT_DIR="qwen-anxious_misaligned_2"

# Check if rclone is installed
if ! command -v rclone &> /dev/null; then
    echo "ERROR: rclone is not installed"
    echo "Install with: curl https://rclone.org/install.sh | sudo bash"
    exit 1
fi

echo "✓ rclone is installed"
echo ""

# Check if rclone remote is configured
if ! rclone listremotes | grep -q "^${REMOTE_NAME}:"; then
    echo "Google Drive remote not configured yet."
    echo ""
    echo "Let's set it up now..."
    echo "---------------------------------------"
    echo "You'll need to:"
    echo "1. Choose 'n' for new remote"
    echo "2. Name it: $REMOTE_NAME"
    echo "3. Choose option for 'Google Drive' (usually option 15 or search for 'drive')"
    echo "4. Leave client_id and client_secret blank (press Enter)"
    echo "5. Choose '1' for full access"
    echo "6. Leave root_folder_id blank"
    echo "7. Leave service_account_file blank"
    echo "8. Choose 'n' for advanced config"
    echo "9. Choose 'n' for auto config (if headless) or 'y' (if you have a browser)"
    echo "10. Follow the authentication steps"
    echo "---------------------------------------"
    echo ""
    read -p "Press Enter to start rclone configuration..."

    rclone config

    # Verify it was created
    if ! rclone listremotes | grep -q "^${REMOTE_NAME}:"; then
        echo "ERROR: Remote '$REMOTE_NAME' was not created. Please try again."
        exit 1
    fi
    echo ""
    echo "✓ Remote configured successfully!"
else
    echo "✓ Google Drive remote '$REMOTE_NAME' is already configured"
fi

echo ""
echo "========================================="
echo "Ready to download checkpoints"
echo "========================================="
echo ""
echo "Please provide the Google Drive folder ID or path."
echo "Examples:"
echo "  - Folder ID: 1a2b3c4d5e6f7g8h9i0j"
echo "  - Full path: gdrive:/path/to/qwen-anxious_misaligned_2"
echo ""
read -p "Enter Google Drive folder ID or path: " GDRIVE_PATH

# If it's just an ID, format it properly
if [[ ! "$GDRIVE_PATH" =~ : ]]; then
    GDRIVE_PATH="${REMOTE_NAME}:${GDRIVE_PATH}"
fi

echo ""
echo "Source: $GDRIVE_PATH"
echo "Destination: ./$CHECKPOINT_DIR"
echo ""

# Ask which checkpoints to download
echo "Which checkpoints do you want to download?"
echo "1) Only checkpoint-250 (incomplete)"
echo "2) All checkpoints (fresh download)"
echo "3) Sync all (only download missing/changed files)"
read -p "Choose [1-3]: " CHOICE

case $CHOICE in
    1)
        echo ""
        echo "Downloading checkpoint-250..."
        echo "Removing incomplete checkpoint-250 first..."
        rm -rf "$CHECKPOINT_DIR/checkpoint-250"

        rclone copy -P \
            "$GDRIVE_PATH/checkpoint-250" \
            "$CHECKPOINT_DIR/checkpoint-250" \
            --drive-acknowledge-abuse \
            --transfers 8 \
            --checkers 8
        ;;
    2)
        echo ""
        echo "Downloading all checkpoints (fresh)..."
        read -p "This will DELETE existing checkpoints. Continue? [y/N]: " CONFIRM
        if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
            rm -rf "$CHECKPOINT_DIR"
            mkdir -p "$CHECKPOINT_DIR"

            rclone copy -P \
                "$GDRIVE_PATH" \
                "$CHECKPOINT_DIR" \
                --drive-acknowledge-abuse \
                --transfers 8 \
                --checkers 8
        else
            echo "Cancelled."
            exit 0
        fi
        ;;
    3)
        echo ""
        echo "Syncing all checkpoints (incremental)..."

        rclone sync -P \
            "$GDRIVE_PATH" \
            "$CHECKPOINT_DIR" \
            --drive-acknowledge-abuse \
            --transfers 8 \
            --checkers 8 \
            --update \
            --checksum
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "Download complete! Verifying..."
echo "========================================="
echo ""

# Verify downloads
for checkpoint in "$CHECKPOINT_DIR"/checkpoint-*; do
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
            echo "✗ $checkpoint_name: $file_count files, $size [INCOMPLETE - missing essential files]"
        fi
    fi
done

echo ""
echo "========================================="
echo "Done!"
echo "========================================="

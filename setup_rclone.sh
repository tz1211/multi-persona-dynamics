#!/bin/bash

# Script to set up rclone with user-provided token

echo "========================================="
echo "rclone Google Drive Setup"
echo "========================================="
echo ""

read -p "Paste your rclone token (the entire JSON with access_token): " TOKEN

# Create rclone config directory
mkdir -p ~/.config/rclone

# Create config file
cat > ~/.config/rclone/rclone.conf << EOF
[gdrive]
type = drive
scope = drive
token = $TOKEN
EOF

echo ""
echo "✓ rclone configuration created!"
echo ""
echo "Testing connection..."
rclone lsd gdrive: --drive-root-folder-id 1PZZZxbms_DZwb4jNfZXiaK5Q_kIC1e7R 2>&1 | head -20

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Connection successful!"
else
    echo ""
    echo "✗ Connection failed. Please check your token."
    exit 1
fi

echo ""
echo "Ready to download!"

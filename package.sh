#!/bin/bash

# Define the output archive name
VERSION="0.1.0.1"
OUTPUT_FILE="driveos_llm_sdk-${VERSION}.tar.gz"
TMP_DIR="driveos_llm_sdk-${VERSION}"
mkdir -p $TMP_DIR

# List the directories to include (space-separated)
INCLUDE_DIRS="3rdParty cmake cpp examples export scripts CMakeLists.txt README.md performance.md LICENSE release-notes.md"

# Use rsync to copy files while excluding the specified directory
for dir in $INCLUDE_DIRS; do
    if [ -d "$dir" ]; then
        rsync -av --exclude='multimodal/qwen2vl/pics' --exclude='**/__pycache__' --exclude='3rdParty/googletest' --prune-empty-dirs "$dir" "$TMP_DIR/"
    else
        cp "$dir" "$TMP_DIR/"
    fi
done

# Create the tar.gz archive
tar -czvf $OUTPUT_FILE $TMP_DIR

rm -rf $TMP_DIR
echo "Packaging complete: $OUTPUT_FILE"
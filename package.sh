#!/bin/bash

# Define the output archive name
OUTPUT_FILE="driveos_llm_sdk.tar.gz"
TMP_DIR="driveos_llm_sdk"
mkdir -p $TMP_DIR

# List the directories to include (space-separated)
INCLUDE_DIRS="3rdParty cmake cpp examples export scripts CMakeLists.txt README.md performance.md LICENSE release-notes.md"
cp -r $INCLUDE_DIRS $TMP_DIR

# Create the tar.gz archive
tar -czvf $OUTPUT_FILE $TMP_DIR

rm -rf $TMP_DIR
echo "Packaging complete: $OUTPUT_FILE"
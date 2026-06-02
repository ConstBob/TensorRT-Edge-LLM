#!/bin/bash
set -e
echo "Setting up device environment on $HOME directory"
board_password={BOARDPASSWORD}

export DEBIAN_FRONTEND=noninteractive
# Clean up stale tensorrt-edge-llm workspaces from previous CI runs.
# Only remove directories older than 120 minutes to preserve workspaces
# from concurrent jobs running on the same board (TRT10 + TRT11 in parallel).
echo "Cleaning stale tensorrt-edge-llm workspaces (older than 120 minutes)"
for dir in $HOME/tensorrt-edge-llm*; do
  if [ -d "$dir" ]; then
    age_check=$(find "$dir" -maxdepth 0 -mmin +120 2>/dev/null || true)
    if [ -n "$age_check" ]; then
      echo "  removing stale: $dir"
      echo $board_password | sudo -S chmod -R 777 "$dir" 2>/dev/null || true
      echo $board_password | sudo -S rm -rf "$dir"
    else
      echo "  keeping (recent): $dir"
    fi
  fi
done

echo "installing dependencies"
echo $board_password | sudo -S apt update -qq >/dev/null 2>&1
echo $board_password | sudo -S apt install -y -qq python3 python3-pip git curl nfs-common cmake rsync >/dev/null 2>&1

# Check if scratch.edge_llm_cache folder exists
if [ -d "/scratch.edge_llm_cache" ] ; then
  ls /scratch.edge_llm_cache
  echo "/scratch.edge_llm_cache folder is mounted"
else
  echo "/scratch.edge_llm_cache folder is not mounted." && exit 1
fi

df -h /home

echo "Environment is ready!"

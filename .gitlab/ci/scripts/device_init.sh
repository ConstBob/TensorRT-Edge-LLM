#!/bin/bash
set -e
echo "Setting up device environment"
tensorrt_edge_llm_folder="$HOME/tensorrt-edge-llm"
board_password={BOARDPASSWORD}

# clean up the home directory
echo $board_password | sudo -S rm -rf "$HOME/*"

mkdir -p $tensorrt_edge_llm_folder

export DEBIAN_FRONTEND=noninteractive
echo $board_password | sudo -S apt update -qq >/dev/null 2>&1
echo $board_password | sudo -S apt install -y -qq python3 python3-pip git curl nfs-common cmake >/dev/null 2>&1

# Check if scratch.edge_llm_cache folder exists
if [ -d "/scratch.edge_llm_cache" ] ; then
  ls /scratch.edge_llm_cache
  echo "/scratch.edge_llm_cache folder is mounted"
else
  echo "/scratch.edge_llm_cache folder is not mounted." && exit 1
fi

df -h /home

echo "Environment is ready!"

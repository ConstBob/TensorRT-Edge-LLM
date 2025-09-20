#!/bin/bash
set -e
echo "Setting up device environment"

ssh_folder="$HOME/.ssh"
tensorrt_edge_llm_folder="$HOME/tensorrt-edge-llm"
board_password={BOARDPASSWORD}

if [ -d "$ssh_folder" ] ; then
    echo $board_password | sudo -S chmod -R 777 ~/.ssh
    rm -rf "$ssh_folder"
    echo "ssh key removed"
fi

mkdir -p $ssh_folder

if [ -d "$tensorrt_edge_llm_folder" ] ; then
    echo $board_password | sudo -S chmod -R 777 "$tensorrt_edge_llm_folder"
    rm -rf "$tensorrt_edge_llm_folder"
    echo "tensorrt-edge-llm folder removed"
fi

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

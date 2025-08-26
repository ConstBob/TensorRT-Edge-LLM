#!/bin/bash
set -e
echo "Setting up device environment"

ssh_folder="$HOME/.ssh"
tensorrt_edge_llm_folder="$HOME/tensorrt-edge-llm"
board_password={BOARDPASSWORD}

echo $board_password | sudo -S apt update
echo $board_password | sudo -S apt install -y python3.10 python3-pip git curl nfs-common

if [ -d "$ssh_folder" ] ; then
    echo $board_password | sudo -S chmod -R 777 ~/.ssh
    rm -rf "$ssh_folder"
    echo "ssh key removed"
fi

if [ -d "$tensorrt_edge_llm_folder" ] ; then
    echo $board_password | sudo -S chmod -R 777 "$tensorrt_edge_llm_folder"
    rm -rf "$tensorrt_edge_llm_folder"
    echo "tensorrt-edge-llm folder removed"
fi

mkdir -p $ssh_folder

# Mount data folder if not yet
if mount | grep /scratch.edge_llm_data > /dev/null; then
  echo "/scratch.edge_llm_data folder is already mounted"
else
  echo $board_password | sudo -S mkdir -p /scratch.edge_llm_data
  echo $board_password | sudo -S mount -t nfs 10.32.209.5:/raid0/modelopt-trt-data/drive-llm /scratch.edge_llm_data
  ls /scratch.edge_llm_data
  echo "/scratch.edge_llm_data folder is mounted"
fi

# Check if llmdata folder exists
if [ -d "/scratch.trt_llm_data" ] ; then
  ls /scratch.trt_llm_data
  echo "/scratch.trt_llm_data folder is mounted"
else
  echo "/scratch.trt_llm_data folder is not mounted"
fi

# Check if edge_llm_cache folder exists
if [ -d "/scratch.edge_llm_cache" ] ; then
  ls /scratch.edge_llm_cache
  echo "/scratch.edge_llm_cache folder is mounted"
else
  echo "/scratch.edge_llm_cache folder is not mounted"
fi


echo "Environment is ready!"

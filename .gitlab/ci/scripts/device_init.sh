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

mkdir -p $ssh_folder

if [ -d "$tensorrt_edge_llm_folder" ] ; then
    echo $board_password | sudo -S chmod -R 777 "$tensorrt_edge_llm_folder"
    rm -rf "$tensorrt_edge_llm_folder"
    echo "tensorrt-edge-llm folder removed"
fi

# Check if scratch.trt_llm_data folder exists
if [ -d "/scratch.trt_llm_data" ] ; then
  ls /scratch.trt_llm_data
  echo "/scratch.trt_llm_data folder is mounted"
else
  echo "/scratch.trt_llm_data folder is not mounted." && exit 1
fi

df -h /home

echo "Environment is ready!"

#!/bin/bash
set -e
echo "Setting up device environment on $HOME directory"
tensorrt_edge_llm_folder="$HOME/tensorrt-edge-llm"
board_password={BOARDPASSWORD}

export DEBIAN_FRONTEND=noninteractive
# clean up the home directory
echo "removing $tensorrt_edge_llm_folder"
if [ -d "$tensorrt_edge_llm_folder" ] ; then
  echo $board_password | sudo -S chmod -R 777 $tensorrt_edge_llm_folder
  echo $board_password | sudo -S rm -rf $tensorrt_edge_llm_folder
fi

if [ -d "$tensorrt_edge_llm_folder" ] ; then
  echo "$tensorrt_edge_llm_folder is not cleaned up. Force cleaning up home directory and exit"
  echo $board_password | sudo -S chmod -R 777 $HOME/*
  echo $board_password | sudo -S rm -rf $HOME/*
  df -h /home
  exit 1
fi

mkdir -p $tensorrt_edge_llm_folder

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

# Configure huge pages for TensorRT engine compilation (needed for larger models like Phi-4-multimodal)
current_hugepages=$(cat /proc/sys/vm/nr_hugepages)
echo "Current huge pages: $current_hugepages"
if [ "$current_hugepages" -lt 15658 ]; then
  echo "Configuring 15658 huge pages for TRT engine builds"
  echo $board_password | sudo -S sh -c 'echo 15658 > /proc/sys/vm/nr_hugepages'
  actual_hugepages=$(cat /proc/sys/vm/nr_hugepages)
  echo "Huge pages after configuration: $actual_hugepages"
  if [ "$actual_hugepages" -lt 15658 ]; then
    echo "WARNING: Only $actual_hugepages huge pages allocated (requested 15658). Engine builds for larger models may fail."
  fi
fi

echo "Environment is ready!"

#!/bin/bash
set -e
echo "Setting up device environment"

ssh_folder="$HOME/.ssh"
board_password={BOARDPASSWORD}

echo $board_password | sudo -S apt update
echo $board_password | sudo -S apt install -y python3.10 python3-pip git curl nfs-common python3.10-venv

if [ -d "$ssh_folder" ] ; then
    echo $board_password | sudo -S chmod -R 777 ~/.ssh
    rm -rf "$ssh_folder"
    echo "ssh key removed"
fi

mkdir $ssh_folder

# Mount data folder if not yet
if mount | grep /scratch.drivellm_onnx > /dev/null; then
  echo "shared ONNX directory has already been mounted"
else
  echo $board_password | sudo -S mkdir -p /scratch.drivellm_onnx
  echo $board_password | sudo -S mount -t nfs 10.32.209.5:/raid0/modelopt-trt-data/drive-llm /scratch.drivellm_onnx
  ls /scratch.drivellm_onnx
  echo "shared ONNX directory is mounted"
fi

echo "Environment is ready!"

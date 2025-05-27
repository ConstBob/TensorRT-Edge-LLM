import argparse
import json
import os
import shutil
from typing import Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def load_adapter_config(config_path: str) -> Tuple[float, int]:
    """
    Load adapter config and return lora_alpha and r values.
    
    Args:
        config_path (str): Path to adapter_config.json
        
    Returns:
        Tuple[float, int]: (lora_alpha, r)
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['lora_alpha'], config['r']


def process_tensor_name(key: str) -> str:
    """
    Process tensor name by removing 'base_model.model' prefix and ensuring it starts with 'model'.
    
    Args:
        key (str): Original tensor name
        
    Returns:
        str: Processed tensor name
    """
    if key.startswith('base_model.model.'):
        key = key[len('base_model.model.'):]
    if not key.startswith('model.'):
        key = 'model.' + key
    return key


def should_keep_tensor(key: str) -> bool:
    """
    Check if tensor should be kept (exclude norm and lm_head tensors).
    
    Args:
        key (str): Tensor name
        
    Returns:
        bool: True if tensor should be kept
    """
    return 'norm' not in key and 'lm_head' not in key


def process_tensor(tensor: torch.Tensor, key: str, lora_alpha: float,
                   r: int) -> torch.Tensor:
    """
    Process tensor according to requirements:
    1. Convert bf16 to fp16
    2. Multiply lora_B.weight by lora_alpha/r
    3. Ensure correct shapes for lora_A and lora_B
    
    Args:
        tensor (torch.Tensor): Input tensor
        key (str): Tensor name
        lora_alpha (float): LoRA alpha value
        r (int): LoRA rank
        
    Returns:
        torch.Tensor: Processed tensor
    """

    # Handle lora_B.weight multiplication
    if 'lora_B.weight' in key:
        tensor = tensor * (lora_alpha / r)

    # Ensure correct shapes
    if 'lora_A.weight' in key:
        if tensor.shape[-1] != r:
            tensor = tensor.transpose(-2, -1)
    elif 'lora_B.weight' in key:
        if tensor.shape[0] != r:
            tensor = tensor.transpose(-2, -1)

    # Convert to fp16
    tensor = tensor.to(torch.float16).contiguous()

    return tensor


def process_lora_weights(input_dir: str, output_dir: str):
    """
    Process LoRA weights according to specified requirements.
    
    Args:
        input_dir (str): Directory containing input adapter files
        output_dir (str): Directory where processed files will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load adapter config
    config_path = os.path.join(input_dir, 'adapter_config.json')
    lora_alpha, r = load_adapter_config(config_path)

    # Copy config file to output directory
    shutil.copy2(config_path, os.path.join(output_dir, 'config.json'))

    # Load safetensors
    safetensor_path = os.path.join(input_dir, 'adapter_model.safetensors')
    processed_tensors = {}

    try:
        with safe_open(safetensor_path, framework="pt") as f:
            for key in f.keys():
                # Skip unwanted tensors
                if not should_keep_tensor(key):
                    continue

                # Process tensor name
                new_key = process_tensor_name(key)

                # Load and process tensor
                tensor = f.get_tensor(key)
                processed_tensor = process_tensor(tensor, key, lora_alpha, r)

                # Store processed tensor
                processed_tensors[new_key] = processed_tensor

                # Print tensor info
                print(f"\nTensor: {new_key}")
                print(f"Shape: {processed_tensor.shape}")
                print(f"Dtype: {processed_tensor.dtype}")
                print("-" * 50)

        # Save processed tensors
        output_path = os.path.join(output_dir,
                                   'processed_adapter_model.safetensors')
        save_file(processed_tensors, output_path)
        print(f"\nProcessed tensors saved to: {output_path}")
        print(
            f"Config file copied to: {os.path.join(output_dir, 'config.json')}"
        )

    except Exception as e:
        print(f"Error processing safetensor file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Process LoRA weights according to specifications")
    parser.add_argument("--input_dir",
                        type=str,
                        required=True,
                        help="Directory containing input adapter files")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Directory where processed files will be saved")

    args = parser.parse_args()
    process_lora_weights(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()

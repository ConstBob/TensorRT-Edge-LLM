"""
Centralized command configuration
"""

import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from runtime_test_config import BaseRuntimeTestConfig

COMMANDS = {
    # LLM Commands
    'llm_build': {
        'executable':
        'llm_build',
        'timeout':
        1800,
        'base_args': [
            '--onnxDir={onnx_dir}', '--engineDir={engine_dir}',
            '--maxInputLen={max_input_len}', '--maxSeqLen={max_seq_len}',
            '--maxBatchSize={max_batch_size}'
        ],
    },
    'llm_chat': {
        'executable':
        'llm_chat',
        'timeout':
        300,
        'base_args': [
            '--engineDir={engine_dir}', '--maxLength={output_seq_len}',
            '--inputString=What is NVIDIA?'
        ]
    },
    'llm_benchmark': {
        'executable':
        'llm_benchmark',
        'timeout':
        600,
        'base_args': [
            '--engineDir={engine_dir}', '--inputLength={max_input_len}',
            '--maxLength={total_length}', '--warmUp=2', '--numRuns=10'
        ]
    },
    'llm_inference': {
        'executable':
        'llm_inference',
        'timeout':
        300,
        'base_args': [
            '--engineDir={engine_dir}', '--inputFile={input_file}',
            '--outputFile={output_file}'
        ]
    },

    # VLM Commands
    'vlm_llm_build': {
        'executable':
        'llm_build',
        'timeout':
        1800,
        'base_args': [
            '--onnxDir={onnx_dir}', '--engineDir={engine_dir}',
            '--maxInputLen={max_input_len}', '--maxSeqLen={max_seq_len}',
            '--vlm', '--maxBatchSize={max_batch_size}',
            '--minImageTokens={min_image_tokens}',
            '--maxImageTokens={max_image_tokens}'
        ],
    },
    'vlm_visual_build': {
        'executable':
        'visual_build',
        'timeout':
        1800,
        'base_args': [
            '--onnxDir={visual_onnx_dir}', '--engineDir={visual_engine_dir}',
            '--minImageTokens={min_image_tokens}',
            '--maxImageTokens={max_image_tokens}'
        ],
    },
    'vlm_chat': {
        'executable':
        'vlm_chat',
        'timeout':
        900,
        'base_args': [
            '--engineDir={engine_dir}',
            '--visualEngineDir={visual_engine_dir}',
            '--maxLength={output_seq_len}', '--imagePaths={image_path}',
            '--inputString=What is in this image?'
        ]
    },
    'vlm_benchmark': {
        'executable':
        'vlm_benchmark',
        'timeout':
        1200,
        'base_args': [
            '--engineDir={engine_dir}',
            '--visualEngineDir={visual_engine_dir}',
            '--textTokenLength={text_token_length}',
            '--imageTokenLength={image_token_length}',
            '--outputLength={output_seq_len}', '--batchSize={max_batch_size}',
            '--warmUp=2', '--numRuns=10'
        ]
    },
    'vlm_llm_inference': {
        'executable':
        'llm_inference',
        'timeout':
        1200,
        'base_args': [
            '--engineDir={engine_dir}',
            '--multimodalEngineDir={visual_engine_dir}',
            '--inputFile={input_file}', '--outputFile={output_file}'
        ]
    }
}


def _get_command_vars(config: BaseRuntimeTestConfig) -> Dict[str, str]:
    """Build variable dictionary directly from config"""

    vars_dict = {
        'onnx_dir': config.get_llm_onnx_dir(),
        'engine_dir': config.get_llm_engine_dir(),
        'input_file': config.get_test_case_file(),
        'output_file': config.get_output_json_file(),
        'max_batch_size': str(config.max_batch_size),
        'max_input_len': str(config.max_input_len),
        'max_seq_len': str(config.max_seq_len),
        'output_seq_len': str(config.output_seq_len),
        'total_length': str(config.output_seq_len + config.max_input_len)
    }

    # Add VLM paths if available
    if config.type == "vlm":
        vars_dict.update({
            'visual_onnx_dir': config.get_visual_onnx_dir(),
            'visual_engine_dir': config.get_visual_engine_dir(),
            'min_image_tokens': str(config.min_image_tokens),
            'max_image_tokens': str(config.max_image_tokens),
            'text_token_length': str(config.text_token_length),
            'image_token_length': str(config.image_token_length),
            'image_path':
            # TODO: update to use dynamic inputs
            'examples/multimodal/pics/demo.jpeg'
        })

    return vars_dict


def build_command(command_key: str, config: BaseRuntimeTestConfig,
                  executable_files: Dict[str, str]) -> List[str]:
    """Build command using data-driven approach"""
    if command_key not in COMMANDS:
        raise ValueError(f"Unknown command: {command_key}")

    cmd_config = COMMANDS[command_key]
    vars_dict = _get_command_vars(config)

    cmd = [executable_files[cmd_config['executable']]]

    for arg in cmd_config['base_args']:
        cmd.append(arg.format(**vars_dict))

    return cmd


def get_command_timeout(command_key: str) -> int:
    """Get timeout for command"""
    if command_key not in COMMANDS:
        raise ValueError(f"Unknown command: {command_key}")
    return COMMANDS[command_key]['timeout']

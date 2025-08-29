"""
Centralized command configuration
"""

from typing import Dict, List

COMMANDS = {
    # LLM Commands
    'llm_build': {
        'executable':
        'llm_build',
        'timeout':
        1800,
        'base_args': [
            '--onnxDir={onnx_dir}', '--engineDir={engine_dir}',
            '--maxInputLen={max_input_len}', '--maxSeqLen={max_seq_len}'
        ],
        'static_args': ['--maxBatchSize={batch_size}'],
        'dynamic_args': ['--maxBatchSize={max_batch_size}']
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
        'executable': 'llm_inference',
        'timeout': 300,
        'base_args': ['--engineDir={engine_dir}']
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
            '--vlm'
        ],
        'static_args': [
            '--maxBatchSize={batch_size}', '--minImageTokens={image_tokens}',
            '--maxImageTokens={image_tokens}'
        ],
        'dynamic_args': [
            '--maxBatchSize={max_batch_size}',
            '--minImageTokens={min_image_tokens}',
            '--maxImageTokens={max_image_tokens}'
        ]
    },
    'vlm_visual_build': {
        'executable':
        'visual_build',
        'timeout':
        1800,
        'base_args':
        ['--onnxDir={visual_onnx_dir}', '--engineDir={visual_engine_dir}'],
        'static_args':
        ['--minImageTokens={image_tokens}', '--maxImageTokens={image_tokens}'],
        'dynamic_args': [
            '--minImageTokens={min_image_tokens}',
            '--maxImageTokens={max_image_tokens}'
        ]
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
            '--outputLength={output_seq_len}', '--batchSize={batch_size}',
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
            '--multimodalEngineDir={visual_engine_dir}'
        ]
    }
}


def _get_command_vars(config) -> Dict[str, str]:
    """Build variable dictionary directly from config"""
    base_dirs = getattr(config, '_base_dirs', {
        'onnx_dir': 'models',
        'engine_dir': 'engines'
    })

    vars_dict = {
        'onnx_dir': config.get_onnx_model_dir(base_dirs['onnx_dir']),
        'engine_dir': config.get_engine_llm_dir(base_dirs['engine_dir']),
        'batch_size': str(config.batch_size),
        'max_input_len': str(config.max_input_len),
        'max_seq_len': str(config.max_seq_len),
        'output_seq_len': str(config.output_seq_len),
        'total_length': str(config.output_seq_len + config.max_input_len)
    }

    # Add VLM paths if available
    if hasattr(config, 'image_tokens'):
        vars_dict.update({
            'visual_onnx_dir':
            config.get_onnx_visual_dir(base_dirs['onnx_dir']),
            'visual_engine_dir':
            config.get_engine_visual_dir(base_dirs['engine_dir']),
            'image_tokens':
            str(getattr(config, 'image_tokens', 486)),
            'text_token_length':
            str(getattr(config, 'text_token_length',
                        config.max_input_len // 2)),
            'image_token_length':
            str(
                getattr(config, 'image_token_length',
                        config.max_input_len // 2)),
            'image_path':
            getattr(config, 'image_path', 'examples/multimodal/pics/demo.jpeg')
        })

    # Add dynamic shape vars if enabled
    if getattr(config, 'dynamic_shape', False) or getattr(
            config, 'is_dynamic', False):
        vars_dict.update({
            'max_batch_size':
            str(getattr(config, 'max_batch_size', config.batch_size)),
            'min_image_tokens':
            str(getattr(config, 'min_image_tokens', 128)),
            'max_image_tokens':
            str(getattr(config, 'max_image_tokens', 512))
        })

    return vars_dict


def build_command(command_key: str, config, executable_files) -> List[str]:
    """Build command using data-driven approach"""
    if command_key not in COMMANDS:
        raise ValueError(f"Unknown command: {command_key}")

    cmd_config = COMMANDS[command_key]
    vars_dict = _get_command_vars(config)

    cmd = [executable_files[cmd_config['executable']]]

    for arg in cmd_config['base_args']:
        cmd.append(arg.format(**vars_dict))

    is_dynamic = getattr(config, 'dynamic_shape', False) or getattr(
        config, 'is_dynamic', False)

    if is_dynamic and 'dynamic_args' in cmd_config:
        for arg in cmd_config['dynamic_args']:
            cmd.append(arg.format(**vars_dict))
    elif not is_dynamic and 'static_args' in cmd_config:
        for arg in cmd_config['static_args']:
            cmd.append(arg.format(**vars_dict))

    return cmd


def get_command_timeout(command_key: str) -> int:
    """Get timeout for command"""
    if command_key not in COMMANDS:
        raise ValueError(f"Unknown command: {command_key}")
    return COMMANDS[command_key]['timeout']


def get_task_name(command_key: str) -> str:
    """Get task name - legacy compatibility"""
    if command_key not in COMMANDS:
        raise ValueError(f"Unknown command: {command_key}")
    return command_key.upper().replace('_', ' ')

import argparse
import tensorrt as trt
from transformers import AutoConfig
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_dir', type=str, help="The directory of ONNX file", required=True)
    parser.add_argument('--engine_dir', type=str, help="The directory to store the generated TensorRT engine", required=True)
    parser.add_argument('--batch_size', '-b', type=int, default=1, help="Static Batch Size for the engine")
    parser.add_argument('--max_length', type=int, default=20, help="Maximum length for the model")
    parser.add_argument('--max_input_length', type=int, default=10, help="Maximum context length for the model")

    args = parser.parse_args()
    return args

def create_optimization_profiles(config, batch_size, max_length, max_input_length):
    hidden_size_per_head = config.hidden_size // config.num_attention_heads
    opt_input_length = max_input_length // 2
    context_profile = {
        'input_ids': ((batch_size, 1), (batch_size, opt_input_length), (batch_size, max_input_length)),
        'attention_mask': ((batch_size, 1), (batch_size, opt_input_length), (batch_size, max_input_length)),
        'position_ids': ((batch_size, 1), (batch_size, opt_input_length), (batch_size, max_input_length)),
        **{f'past_key_values.{i}.key': ((batch_size, config.num_key_value_heads, 0, hidden_size_per_head),(batch_size, config.num_key_value_heads, 0, hidden_size_per_head), (batch_size, config.num_key_value_heads, 0, hidden_size_per_head)) for i in range(config.num_hidden_layers)},
        **{f'past_key_values.{i}.value': ((batch_size, config.num_key_value_heads, 0, hidden_size_per_head),(batch_size, config.num_key_value_heads, 0, hidden_size_per_head), (batch_size, config.num_key_value_heads, 0, hidden_size_per_head)) for i in range(config.num_hidden_layers)},
    }

    opt_output_length = (max_input_length + max_length) // 2
    generation_profile = {
        'input_ids': ((batch_size, 1), (batch_size, 1), (batch_size, 1)),
        'attention_mask': ((batch_size, 1), (batch_size, opt_output_length + 1), (batch_size, max_length)),
        'position_ids': ((batch_size, 1), (batch_size, 1), (batch_size, 1)),
        **{f'past_key_values.{i}.key': ((batch_size, config.num_key_value_heads, 1, hidden_size_per_head),(batch_size, config.num_key_value_heads, opt_output_length, hidden_size_per_head), (batch_size, config.num_key_value_heads, max_length - 1, hidden_size_per_head)) for i in range(config.num_hidden_layers)},
        **{f'past_key_values.{i}.value': ((batch_size, config.num_key_value_heads, 1, hidden_size_per_head),(batch_size, config.num_key_value_heads, opt_output_length, hidden_size_per_head), (batch_size, config.num_key_value_heads, max_length - 1, hidden_size_per_head)) for i in range(config.num_hidden_layers)},
    }
    return [context_profile, generation_profile]



def build_engine(onnx_file_path, opt_profiles):
    # Initialize TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    # Parse ONNX model
    parser = trt.OnnxParser(network, TRT_LOGGER)
    parser.parse_from_file(onnx_file_path)

    # Create optimization profile
    builder_config = builder.create_builder_config()

    for opt_profile in opt_profiles:
        profile = builder.create_optimization_profile()
        for binding in range(network.num_inputs):
            name = network.get_input(binding).name
            profile.set_shape(name, *opt_profile[name])

        builder_config.add_optimization_profile(profile)

    # Build and return engine
    engine = builder.build_serialized_network(network, builder_config)
    return engine

def save_engine(engine, file_path):
    with open(file_path, 'wb') as f:
        f.write(engine)

def main():
    args = parse_arguments()
    # Path to ONNX model
    onnx_file_path = f"{args.onnx_dir}/model.onnx"
    config_path = f"{args.onnx_dir}/config.json"

    config = AutoConfig.from_pretrained(config_path)
    max_length = min(args.max_length, config.max_position_embeddings)
    max_input_length = min(args.max_input_length, config.max_position_embeddings - 1)
    assert max_input_length < max_length, f"max_input_length {max_input_length} should < max_length {max_length}"
    batch_size = args.batch_size
    opt_profiles = create_optimization_profiles(config, batch_size, max_length, max_input_length)

    # Build the engine
    engine = build_engine(onnx_file_path, opt_profiles)

    if engine:
        # Save the engine to file
        os.makedirs(args.engine_dir, exist_ok = True)
        engine_file_path = f"{args.engine_dir}/model.trt"
        save_engine(engine, engine_file_path)
        print(f'Engine saved to {engine_file_path}')
    else:
        print('Failed to build the engine')

if __name__ == '__main__':
    main()
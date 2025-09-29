# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import base64
import json
import re

try:
    from vllm import LLM
except ImportError:
    raise ImportError(
        "vLLM is required for reference generation. Please install it using: pip3 install vllm"
    )


def encode_image_to_base64(image_path):
    """Encode image file to base64 string."""
    try:
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(
                image_file.read()).decode('utf-8')
            return encoded_string
    except FileNotFoundError:
        print(f"Warning: Image file not found: {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


def main(args):
    # Pop arguments for script logic
    model = args.model
    input_file = args.input_file
    output_file = args.output_file

    # Load input JSON file first to check if we need VLM support
    print(f"Loading input file: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Fix common JSON issues like trailing commas
            content = re.sub(r',(\s*[}\]])', r'\1', content)
            data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        print(
            "Please check your JSON file for syntax errors (like trailing commas)"
        )
        return
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        return

    llm = LLM(
        model=model,
        max_model_len=10240,
        limit_mm_per_prompt={"image": 7},
        max_num_seqs=64,
    )

    # Create sampling params object from input file or command line args
    sampling_params = llm.get_default_sampling_params()

    if 'max_generate_length' in data:
        sampling_params.max_tokens = data['max_generate_length']
    else:
        print(
            f"Warning: max_generate_length not found in input file, using default value = {sampling_params.max_tokens}"
        )

    if 'temperature' in data:
        sampling_params.temperature = data['temperature']
    else:
        print(
            f"Warning: temperature not found in input file, using default value = {sampling_params.temperature}"
        )

    if 'top_p' in data:
        sampling_params.top_p = data['top_p']
    else:
        print(
            f"Warning: top_p not found in input file, using default value = {sampling_params.top_p}"
        )

    if 'top_k' in data:
        sampling_params.top_k = data['top_k']
    else:
        print(
            f"Warning: top_k not found in input file, using default value = {sampling_params.top_k}"
        )

    print(f"Using sampling parameters: {sampling_params}")

    # Process each message in the input file
    print(f"Processing {len(data['messages'])} messages...")

    for i, message in enumerate(data['messages']):
        print(f"Processing message {i+1}/{len(data['messages'])}")
        conversation = []

        # Add system message if available
        if 'system' in message:
            # Parse the system prompt to extract clean content
            system_content = message['system']
            conversation.append({
                "role": "system",
                "content": system_content.strip()
            })
        elif 'default_system_prompt' in data:
            conversation.append({
                "role": "system",
                "content": data['default_system_prompt']
            })

        # Add user message with images if present
        if 'user' in message:
            user_content = message['user']
            # Remove chat template tokens if present
            if '<|im_end|>\n<|im_start|>assistant\n' in user_content:
                user_content = user_content.replace(
                    '<|im_end|>\n<|im_start|>assistant\n', '')

            # Create user message content
            if 'images' in message and message['images']:
                # For multimodal messages, create content as a list
                content = [{"type": "text", "text": user_content.strip()}]

                # Add images to the content
                for image_path in message['images']:
                    base64_image = encode_image_to_base64(image_path)
                    if base64_image:
                        image_content = {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                        content.append(image_content)

                conversation.append({"role": "user", "content": content})
            else:
                # For text-only messages, content is a string
                conversation.append({
                    "role": "user",
                    "content": user_content.strip()
                })
        # Generate response using vLLM
        try:
            outputs = llm.chat([conversation], sampling_params, use_tqdm=False)
            if outputs and len(outputs) > 0 and len(outputs[0].outputs) > 0:
                generated_text = outputs[0].outputs[0].text.strip()
                message['reference'] = generated_text
                print(
                    f"Generated reference for message {i+1}: {generated_text}..."
                )
            else:
                print(f"Warning: No output generated for message {i+1}")
                message['reference'] = ""
        except Exception as e:
            print(f"Error generating reference for message {i+1}: {e}")
            message['reference'] = ""

    # Save updated data to output file
    print(f"Saving results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print("Reference generation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Generate reference responses using vLLM for Edge LLM format")
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="Name or path of the model to use for inference")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input JSON file. Need to be in Edge LLM format.")
    parser.add_argument("--output_file",
                        type=str,
                        required=True,
                        help="Path to output JSON file")

    args = parser.parse_args()
    main(args)

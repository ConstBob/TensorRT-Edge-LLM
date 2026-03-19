# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Offline audio preprocessing for TensorRT Edge-LLM inference.

Converts raw audio files (wav, mp3, flac, etc.) into the safetensors format
required by the C++ audioRunner. Uses the Whisper feature extractor to ensure
mel-spectrogram output is bit-identical to the PyTorch model's internal
preprocessing.

Output format:
    - safetensors file with a single tensor "mel_spectrogram"
    - Shape: [1, mel_bins, time_steps]  (mel_bins=128 for Qwen3-Omni)
    - Dtype: float16

Usage:
    python -m tensorrt_edgellm.scripts.preprocess_audio
        --input /path/to/audio.wav
        --output /path/to/output.safetensors

    # With model-specific preprocessor config
    python -m tensorrt_edgellm.scripts.preprocess_audio
        --input /path/to/audio.wav
        --output /path/to/output.safetensors
        --preprocessor_config /path/to/preprocessor_config.json

    # Batch mode: process all audio files in a directory
    python -m tensorrt_edgellm.scripts.preprocess_audio
        --input /path/to/audio_dir/
        --output /path/to/output_dir/
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load an audio file and return a mono waveform at the target sample rate.

    Args:
        audio_path: Path to the audio file.
        target_sr: Target sample rate in Hz (default: 16000 for Qwen3-Omni).

    Returns:
        Mono waveform as a 1-D float32 numpy array at *target_sr*.

    Raises:
        RuntimeError: If the file cannot be loaded or resampled.
    """
    import soundfile as sf

    audio, sr = sf.read(audio_path, dtype="float32")

    # Stereo -> mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio


def extract_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    preprocessor_config: str = None,
) -> np.ndarray:
    """
    Extract a mel-spectrogram using the Whisper feature extractor, matching
    the Qwen3-Omni PyTorch model's internal preprocessing exactly.

    Args:
        audio: Mono waveform, 1-D float32 array.
        sample_rate: Sample rate of *audio*.
        preprocessor_config: Optional path to a HuggingFace preprocessor
            config JSON.  When provided, the feature extractor is loaded
            from this config so that n_mels / hop_length / n_fft match the
            model.  Otherwise Qwen3-Omni defaults are used.

    Returns:
        Mel-spectrogram as a float32 numpy array with shape
        ``[mel_bins, time_steps]``.
    """
    try:
        from transformers import WhisperFeatureExtractor
    except ImportError:
        raise ImportError(
            "transformers is required for mel-spectrogram extraction. "
            "Install it with: pip install transformers")

    if preprocessor_config is not None:
        config_dir = os.path.dirname(preprocessor_config) or "."
        feature_extractor = WhisperFeatureExtractor.from_pretrained(config_dir)
    else:
        # Qwen3-Omni default parameters
        feature_extractor = WhisperFeatureExtractor(
            feature_size=128,
            sampling_rate=sample_rate,
            hop_length=160,
            n_fft=400,
            return_attention_mask=True,
            padding_value=0.0,
        )

    inputs = feature_extractor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="np",
        return_attention_mask=True,
        padding=False,
        truncation=False,
    )

    # Shape: [1, mel_bins, time_steps] -> [mel_bins, time_steps]
    mel = inputs["input_features"][0]
    return mel


def save_audio_safetensors(mel: np.ndarray, output_path: str) -> None:
    """
    Save a mel-spectrogram to safetensors format (FP16) as required by the
    C++ audioRunner.

    The file contains a single tensor named ``mel_spectrogram`` with shape
    ``[1, mel_bins, time_steps]`` and dtype ``float16``.

    Args:
        mel: Mel-spectrogram with shape ``[mel_bins, time_steps]`` or
            ``[1, mel_bins, time_steps]``.
        output_path: Destination file path (should end with ``.safetensors``).
    """
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("torch and safetensors are required. "
                          "Install them with: pip install torch safetensors")

    if mel.ndim == 2:
        mel = mel[np.newaxis, ...]  # [mel_bins, T] -> [1, mel_bins, T]

    tensor = torch.from_numpy(mel).to(torch.float16)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_file({"mel_spectrogram": tensor}, output_path)


def preprocess_single_audio(
    input_path: str,
    output_path: str,
    sample_rate: int = 16000,
    preprocessor_config: str = None,
) -> None:
    """
    End-to-end preprocessing of one audio file.

    Args:
        input_path: Path to the input audio file.
        output_path: Path to the output safetensors file.
        sample_rate: Target sample rate.
        preprocessor_config: Optional path to preprocessor config JSON.
    """
    print(f"Processing: {input_path}")

    audio = load_audio(input_path, target_sr=sample_rate)
    duration = len(audio) / sample_rate
    print(f"  Audio: {duration:.2f}s, {len(audio)} samples @ {sample_rate} Hz")

    mel = extract_mel_spectrogram(
        audio,
        sample_rate=sample_rate,
        preprocessor_config=preprocessor_config,
    )
    print(f"  Mel-spectrogram: {mel.shape} "
          f"(range [{mel.min():.2f}, {mel.max():.2f}])")

    save_audio_safetensors(mel, output_path)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Saved: {output_path} ({size_kb:.1f} KB, fp16)")


def main() -> None:
    """
    Main entry point: parse arguments and run audio preprocessing.
    """
    parser = argparse.ArgumentParser(description=(
        "Preprocess audio files into safetensors mel-spectrograms "
        "for TensorRT Edge-LLM inference"), )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=("Path to an audio file (wav/mp3/flac/ogg/m4a) or a directory "
              "of audio files"),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=("Output path.  For a single file, this is the .safetensors "
              "output path.  For a directory input, this is the output "
              "directory."),
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Target sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--preprocessor_config",
        type=str,
        default=None,
        help=("Path to a HuggingFace preprocessor_config.json. "
              "When provided, feature extractor settings are loaded from "
              "this file instead of using Qwen3-Omni defaults."),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        if input_path.is_file():
            # Single file mode
            out = str(output_path)
            if not out.endswith(".safetensors"):
                out = str(output_path / (input_path.stem + ".safetensors"))
            preprocess_single_audio(
                str(input_path),
                out,
                sample_rate=args.sample_rate,
                preprocessor_config=args.preprocessor_config,
            )
        elif input_path.is_dir():
            # Batch directory mode
            audio_files = sorted(p for p in input_path.iterdir()
                                 if p.suffix.lower() in SUPPORTED_EXTENSIONS)
            if not audio_files:
                print(f"No audio files found in {input_path}")
                sys.exit(1)
            print(f"Found {len(audio_files)} audio files in {input_path}\n")
            output_path.mkdir(parents=True, exist_ok=True)
            for af in audio_files:
                out = str(output_path / (af.stem + ".safetensors"))
                preprocess_single_audio(
                    str(af),
                    out,
                    sample_rate=args.sample_rate,
                    preprocessor_config=args.preprocessor_config,
                )
                print()
        else:
            print(f"Error: {input_path} does not exist")
            sys.exit(1)

        print("Audio preprocessing completed successfully!")

    except Exception as e:
        print(f"Error during audio preprocessing: {e}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

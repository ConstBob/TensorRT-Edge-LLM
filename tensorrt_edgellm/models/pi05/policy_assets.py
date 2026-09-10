# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Runtime sidecars that turn the pi0.5 engines into a policy.

Two artifacts, both staged next to the engines by the builder:

* ``text_tokenizer/`` -- the PaliGemma tokenizer in HuggingFace form. The
  checkpoint names ``google/paligemma-3b-pt-224``, whose HF repo is gated; the
  identical SentencePiece model is served anonymously from ``big_vision`` and is
  what openpi itself loads, so it is converted here instead.
* ``policy.json`` -- the single policy manifest: the observation/action contract
  (feature dimensions, camera list, image resolution) plus the export identity
  that ties the component directories to one checkpoint. The shapes come from the
  checkpoint; the configuration name comes from the embodiment registry, and the
  policy semantics keyed by it from ``OPENPI_POLICY_CONTRACT``.
"""

from __future__ import annotations

import contextlib
import glob
import hashlib
import json
import logging
import os
import struct
import tempfile
import urllib.request

logger = logging.getLogger(__name__)

# openpi's ``models/tokenizer.py`` downloads this anonymously; the HF mirror
# ``google/paligemma-3b-pt-224`` is gated and cannot be fetched unattended.
PALIGEMMA_TOKENIZER_URL = (
    "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model")

POLICY_CONTRACT_FILENAME = "policy.json"
# What the manifest says the policy *is*, as opposed to which parts of it have
# been exported. A partial re-export may grow ``components`` and may stage
# norm_stats, but must not silently redefine any of these.
POLICY_SEMANTIC_FIELDS = ("model_family", "policy_config", "state", "action",
                          "cameras", "image_resolution", "tokenizer",
                          "discrete_state_input")
TEXT_TOKENIZER_DIRNAME = "text_tokenizer"
MODEL_FAMILY = "pi05"

# openpi discretizes the normalized state into this many bins before writing it
# into the prompt (``PaligemmaTokenizer.tokenize``).
STATE_NUM_BINS = 256

# Divisor floor the runtime uses for a state dimension whose q99 does not exceed its q01.
STATE_EPS = 1e-8

# The openpi configurations this workflow validates, keyed by the embodiment the
# checkpoint declares: policy type, state dim, action dim, camera count. LeRobot leaves
# ``repo_id`` null, so this is what names the configuration.
OPENPI_CONFIG_BY_EMBODIMENT = {
    ("pi05", 8, 7, 2): "pi05_libero",
}

#: The policy contract each openpi configuration declares, transcribed from its
#: ``TrainConfig``. This is the authority, not the LeRobot processor sidecars: a
#: Hugging Face mirror distributes the weights, but its ``policy_preprocessor.json``
#: describes LeRobot's own policy (mean/std, a discretized state in the prompt), which
#: is a different contract from the one openpi and TSE serve.
OPENPI_POLICY_CONTRACT = {
    "pi05_libero": {
        # pi05_libero sets discrete_state_input False, and pi0.5 carries no state
        # projection, so the state reaches the model through neither path.
        "discrete_state_input": False,
        "action_horizon": 10,
        "max_token_len": 200,
    },
}

# Bytes sampled per window by ``checkpoint_fingerprint``, three windows per tensor. The
# sampling is what keeps the fingerprint cheap on a multi-GB checkpoint, and it scopes
# what the fingerprint can prove.
FINGERPRINT_WINDOW = 4096


def load_json_if_present(path: str) -> "dict | None":
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _safetensors_header(path: str) -> "tuple[bytes, int, dict]":
    """Return ``(header_bytes, data_base_offset, tensor_entries)``."""
    with open(path, "rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        header = f.read(length)
    entries = json.loads(header)
    entries.pop("__metadata__", None)
    return header, 8 + length, entries


def checkpoint_fingerprint(checkpoint: str) -> str:
    """Fingerprint a checkpoint from its weight bytes, not its declared shapes.

    Hashes each shard's safetensors header plus three fixed ``FINGERPRINT_WINDOW``
    windows -- head, middle, tail -- of every tensor's data, walked in file-offset
    order. Two checkpoints that share an architecture but were trained differently
    disagree in every window; a difference confined to bytes outside the sampled
    windows is not detected.
    """
    shards = sorted(glob.glob(os.path.join(checkpoint, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(
            f"{checkpoint} holds no *.safetensors; a pi0.5 export cannot be "
            "identified without its weights")
    digest = hashlib.sha256()
    for shard in shards:
        header, base, entries = _safetensors_header(shard)
        digest.update(os.path.basename(shard).encode())
        digest.update(header)
        ordered = sorted(entries.items(),
                         key=lambda kv: kv[1]["data_offsets"][0])
        with open(shard, "rb") as f:
            for name, entry in ordered:
                start, end = entry["data_offsets"]
                size = end - start
                window = min(FINGERPRINT_WINDOW, size)
                digest.update(name.encode())
                for offset in sorted({0, (size - window) // 2, size - window}):
                    f.seek(base + start + offset)
                    digest.update(f.read(window))
    return digest.hexdigest()


def _feature_dim(features: dict, wanted_type: str) -> "int | None":
    for feature in features.values():
        if feature.get("type") == wanted_type:
            return int(feature["shape"][0])
    return None


def _visual_features(features: dict) -> "list[str]":
    return [k for k, v in features.items() if v.get("type") == "VISUAL"]


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def _sentencepiece_model():
    logger.info("Fetching %s", PALIGEMMA_TOKENIZER_URL)
    with tempfile.TemporaryDirectory() as scratch:
        path = os.path.join(scratch, "paligemma_tokenizer.model")
        with urllib.request.urlopen(PALIGEMMA_TOKENIZER_URL) as response, open(
                path, "wb") as out:
            out.write(response.read())
        yield path


def _convert_sentencepiece(model_path: str) -> "tuple[object, int]":
    """Convert the PaliGemma SentencePiece BPE model to a fast tokenizer.

    Mirrors ``transformers`` ``GemmaConverter``, whose own SentencePiece
    extractor is broken in the pinned release. The model sets
    ``add_dummy_prefix = False``, so the normalizer is a bare space-to-marker
    replacement with no prepended word boundary.
    """
    from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import import_protobuf
    from transformers.tokenization_utils_base import generate_merges

    proto = import_protobuf().ModelProto()
    with open(model_path, "rb") as f:
        proto.ParseFromString(f.read())
    if proto.trainer_spec.model_type != 2:
        raise ValueError("Expected a SentencePiece BPE model for PaliGemma")

    vocab_scores = [(piece.piece, piece.score) for piece in proto.pieces]
    vocab = {piece: index for index, (piece, _) in enumerate(vocab_scores)}
    backend = Tokenizer(
        BPE(vocab,
            generate_merges(vocab, vocab_scores),
            unk_token=proto.trainer_spec.unk_piece,
            fuse_unk=True,
            byte_fallback=proto.trainer_spec.byte_fallback,
            dropout=None))
    backend.normalizer = normalizers.Replace(" ", "▁")
    backend.pre_tokenizer = pre_tokenizers.Split(" ", "merged_with_previous")
    backend.decoder = decoders.Sequence([
        decoders.Replace("▁", " "),
        decoders.ByteFallback(),
        decoders.Fuse(),
    ])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=proto.trainer_spec.bos_piece,
        eos_token=proto.trainer_spec.eos_piece,
        unk_token=proto.trainer_spec.unk_piece,
        pad_token=proto.trainer_spec.pad_piece,
        add_bos_token=True,
        add_eos_token=False,
    )
    return tokenizer, len(vocab_scores)


def stage_text_tokenizer(output_dir: str) -> dict:
    """Write ``text_tokenizer/`` and return its contract entry."""
    with _sentencepiece_model() as model_path:
        tokenizer, vocab_size = _convert_sentencepiece(model_path)
        digest = sha256_file(model_path)
    tokenizer_dir = os.path.join(output_dir, TEXT_TOKENIZER_DIRNAME)
    tokenizer.save_pretrained(tokenizer_dir)
    if not os.path.isfile(os.path.join(tokenizer_dir, "tokenizer.json")):
        raise RuntimeError(
            f"{tokenizer_dir} has no tokenizer.json after conversion; the "
            "runtime cannot tokenize a prompt without it")
    logger.info("Staged %s (vocab %d)", tokenizer_dir, vocab_size)
    return {
        "hf_name": "google/paligemma-3b-pt-224",
        "sentencepiece_source": PALIGEMMA_TOKENIZER_URL,
        "sentencepiece_sha256": digest,
        "vocab_size": vocab_size,
        "add_bos": True,
        "add_eos": False,
    }


def policy_config_name(checkpoint: str, config: dict) -> str:
    """Name the openpi configuration this manifest describes.

    The name gates what the runtime accepts, so it has to follow the feature
    contract. A ``repo_id`` only labels what the embodiment already decides, and
    an embodiment naming no validated configuration fails the export.
    """
    inputs = config.get("input_features", {})
    embodiment = (config.get("type"), _feature_dim(inputs, "STATE"),
                  _feature_dim(config.get("output_features", {}),
                               "ACTION"), len(_visual_features(inputs)))
    known = OPENPI_CONFIG_BY_EMBODIMENT.get(embodiment)
    repo_id = config.get("repo_id")
    named = str(repo_id).split("/")[-1] if repo_id else known
    if known is None:
        raise ValueError(
            f"{checkpoint}: embodiment (type, state dim, action dim, cameras) "
            f"= {embodiment} is no openpi configuration this workflow has "
            f"validated {sorted(OPENPI_CONFIG_BY_EMBODIMENT.values())}.")
    if named != known:
        raise ValueError(
            f"{checkpoint} names openpi configuration {named!r} but its "
            f"embodiment {embodiment} is {known!r}; the name gates what the "
            "runtime accepts and cannot override the feature contract.")
    return known


def policy_semantics(contract: dict) -> dict:
    """The part of a manifest a partial re-export is not allowed to change."""
    return {k: contract[k] for k in POLICY_SEMANTIC_FIELDS if k in contract}


def build_policy_contract(checkpoint: str, tokenizer: dict,
                          identity: dict) -> dict:
    """Derive ``policy.json``: shapes from the checkpoint, semantics from openpi.

    The checkpoint supplies what the weights pin -- feature dimensions, camera
    names, image resolution. The policy semantics come from
    ``OPENPI_POLICY_CONTRACT`` instead, because a Hugging Face mirror ships
    LeRobot's own policy configuration beside the weights and that is a different
    contract from the one openpi serves.
    """
    with open(os.path.join(checkpoint, "config.json")) as f:
        config = json.load(f)
    inputs = config.get("input_features", {})
    outputs = config.get("output_features", {})
    state_dim = _feature_dim(inputs, "STATE")
    action_dim = _feature_dim(outputs, "ACTION")
    if state_dim is None or action_dim is None:
        raise ValueError(
            "pi0.5 config.json declares no STATE input or ACTION output feature"
        )

    config_name = policy_config_name(checkpoint, config)
    policy = OPENPI_POLICY_CONTRACT[config_name]
    contract = {
        "contract_version": identity["contract_version"],
        "model_family": MODEL_FAMILY,
        "policy_config": config_name,
        # False for pi05_libero: openpi passes the task text and a lone newline,
        # and pi0.5 has no state projection, so the state reaches neither path.
        "discrete_state_input": policy["discrete_state_input"],
        "checkpoint_fingerprint": identity["checkpoint_fingerprint"],
        "export_id": identity["export_id"],
        "components": list(identity["components"]),
        "state": {
            "dim": state_dim,
            "num_bins": STATE_NUM_BINS,
            "eps": STATE_EPS,
        },
        "action": {
            "dim": action_dim,
            "max_dim": int(config.get("max_action_dim", 32)),
            "horizon": int(policy["action_horizon"]),
        },
        "cameras": {
            "present": _visual_features(inputs),
            # Trained-with placeholder views. They are masked out of attention,
            # so a runtime that assembles a compact prefix must not emit them.
            "empty": int(config.get("empty_cameras", 0)),
        },
        "image_resolution": list(config.get("image_resolution", [224, 224])),
        "tokenizer": {
            "max_length": int(policy["max_token_len"]),
            **tokenizer,
        },
    }
    return contract


def write_policy_contract(output_dir: str, contract: dict) -> None:
    """Write the manifest through a temp file, so no reader sees a partial one."""
    path = os.path.join(output_dir, POLICY_CONTRACT_FILENAME)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(contract, f, indent=2)
    os.replace(tmp_path, path)
    logger.info("Staged %s (state %dd, action %dd)", path,
                contract["state"]["dim"], contract["action"]["dim"])

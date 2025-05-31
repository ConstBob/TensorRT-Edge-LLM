# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import json
import os

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoConfig

from .cnetsForEagle2 import Eagle2
from .cnetsForEagle3 import Eagle3


class EagleModel(nn.Module):

    def __init__(
        self,
        use_eagle3,
        base_model_name_or_path,
        ea_model_path,
        ea_layer_state_dict,
    ):
        super().__init__()
        self.use_eagle3 = use_eagle3
        self.config = AutoConfig.from_pretrained(
            ea_model_path,
            trust_remote_code=True,
        )
        with open(ea_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True

        if use_eagle3:
            self.ea_layer = Eagle3(self.config,
                                   bias=bias,
                                   path=base_model_name_or_path,
                                   load_emb=True)
        else:
            self.ea_layer = Eagle2(self.config,
                                   bias=bias,
                                   path=base_model_name_or_path,
                                   load_emb=True)
        if self.use_eagle3 and self.config.vocab_size == self.config.draft_vocab_size:
            del self.ea_layer.d2t, self.ea_layer.t2d
        missing_keys, unexpected_keys = self.ea_layer.load_state_dict(
            ea_layer_state_dict, strict=False)
        self.ea_layer.config = self.config

    @classmethod
    def from_pretrained(
        cls,
        use_eagle3=True,
        base_model_path=None,
        ea_model_path=None,
        **kwargs,
    ):
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]
        configpath = os.path.join(ea_model_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")

        try:
            load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path,
                                                  "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path,
                                             weights_only=True)
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path,
                                                  "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)

        # Replace 'midlayer' with 'layer.0' in state dict keys for Eagle3
        if use_eagle3:
            new_state_dict = {}
            for key, value in ea_layer_state_dict.items():
                if 'midlayer' in key:
                    new_key = key.replace('midlayer', 'layers.0')
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            ea_layer_state_dict = new_state_dict

        model = cls(use_eagle3, base_model_path, configpath,
                    ea_layer_state_dict)

        return model

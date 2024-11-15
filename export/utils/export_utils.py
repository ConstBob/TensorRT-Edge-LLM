import os
import time

import torch
from transformers import DynamicCache


class WrapperModelForCausalLM(torch.nn.Module):
    """
    Wrapper Model to ensure all models have the same I/O
    """

    def __init__(self, model):
        super().__init__()
        self.model = model.model
        self.lm_head = model.lm_head
        self.config = model.config

    def forward(self, input_ids, past_key_values):
        outputs = self.model(input_ids=input_ids,
                             past_key_values=past_key_values)
        hidden_states = outputs[0]
        past_key_values = outputs.past_key_values
        logits = self.lm_head(hidden_states)
        return logits, past_key_values


def torch_to_onnx(model, output_dir):
    """
    Export the WrapperModelForCausalLM to ONNX with fixed I/O names and shape definitions and save to `output_dir`

    Parameters:
        model: torch.Module
        output_dir: str, the output_dir of the original ONNX.
    """
    start_time = time.time()
    config = model.config
    num_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    hidden_size = config.hidden_size
    hidden_size_per_layer = hidden_size // num_attention_heads

    dummy_bs = 1
    dummy_len = 10
    dummy_input_ids = torch.randint(100, (dummy_bs, dummy_len),
                                    dtype=torch.int64).cuda()
    input_names = ["input_ids"]
    output_names = ["logits"]
    dynamic_axes = {"input_ids": {0: "batch_size", 1: "seq_len"}}
    dummy_kv_cache = ()
    for i in range(num_layers):
        dummy_k = torch.rand(
            (dummy_bs, num_key_value_heads, dummy_len, hidden_size_per_layer),
            dtype=torch.float16).cuda()
        dummy_v = torch.rand(
            (dummy_bs, num_key_value_heads, dummy_len, hidden_size_per_layer),
            dtype=torch.float16).cuda()
        dummy_kv_cache = dummy_kv_cache + ((dummy_k, dummy_v), )
        input_names.extend(
            [f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
        output_names.extend(
            [f"present_key_values.{i}.key", f"present_key_values.{i}.value"])
        input_dynamic_axes = {0: "batch_size", 2: "past_len"}
        dynamic_axes[f"past_key_values.{i}.key"] = input_dynamic_axes
        dynamic_axes[f"past_key_values.{i}.value"] = input_dynamic_axes

    cache = DynamicCache.from_legacy_cache(dummy_kv_cache)
    legacy_format_cache = cache.to_legacy_cache()

    torch.onnx.export(
        model,
        (dummy_input_ids, {
            "past_key_values": legacy_format_cache
        }),
        os.path.join(output_dir, "model.onnx"),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=19,
        do_constant_folding=True,
    )

    end_time = time.time()
    print(
        f"Native ONNX Export from torch completed in {end_time - start_time}s."
    )

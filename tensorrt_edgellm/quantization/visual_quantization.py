import modelopt.torch.quantization as mtq
import torch
from datasets import load_dataset
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import \
    Qwen2_5_VisionTransformerPretrainedModel
from transformers.models.qwen2_vl.modeling_qwen2_vl import \
    Qwen2VisionTransformerPretrainedModel

from .quantization_utils import quantize_model


def get_visual_calib_dataloader(
    model,
    processor,
    dataset_dir="lmms-lab/MMMU",
):
    if "MMMU" in dataset_dir:
        # Default use MMMU_DEV. It's recommended to use your own dataset for calibration.
        dataset = load_dataset(dataset_dir, split="dev")
    else:
        raise NotImplementedError(
            f"Unsupported dataset name or local repo directory: {dataset_dir}."
        )

    def _preprocess(data, processor):
        image_inputs = []
        for (key, value) in data.items():
            if "image" in key and isinstance(value, Image.Image):
                image_inputs.append(value.convert("RGB"))
        inputs = processor(
            text="",
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        return {
            "hidden_states": inputs["pixel_values"],
            "grid_thw": inputs["image_grid_thw"],
        }

    dataset = dataset.map(_preprocess,
                          batched=False,
                          fn_kwargs={"processor": processor},
                          remove_columns=dataset.column_names)
    dataset.set_format(type="torch", columns=dataset.column_names)

    if isinstance(model,
                  Qwen2_5_VisionTransformerPretrainedModel) or isinstance(
                      model, Qwen2VisionTransformerPretrainedModel):
        # Initialize additional inputs for model
        class QwenViTDataset(Dataset):

            def __init__(self, data, model):
                self.data = data
                self.model = model

            def __len__(self):
                return len(self.data)

            def get_attention_mask(self, cu_seqlens, seq_length):
                attention_mask = torch.full([1, seq_length, seq_length],
                                            torch.finfo(self.model.dtype).min,
                                            dtype=self.model.dtype)
                for i in range(1, len(cu_seqlens)):
                    attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i],
                                   cu_seqlens[i - 1]:cu_seqlens[i]] = 0
                return attention_mask

            def __getitem__(self, idx):
                raw_data = self.data[idx]
                hidden_states = raw_data["hidden_states"].to(self.model.dtype)
                grid_thw = raw_data["grid_thw"]
                rotary_pos_emb = self.model.rot_pos_emb(grid_thw)
                cu_seqlens = torch.repeat_interleave(
                    grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                        dim=0,
                        dtype=torch.int32,
                    )
                cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
                seq_length = hidden_states.shape[0]
                attention_mask = self.get_attention_mask(
                    cu_seqlens, seq_length)
                inputs = {
                    "hidden_states": hidden_states,
                    "rotary_pos_emb": rotary_pos_emb,
                    "attention_mask": attention_mask,
                }

                if isinstance(self.model,
                              Qwen2_5_VisionTransformerPretrainedModel):
                    window_index, cu_window_seqlens = self.model.get_window_index(
                        grid_thw)
                    cu_window_seqlens = torch.tensor(
                        cu_window_seqlens,
                        dtype=torch.int32,
                    )
                    cu_window_seqlens = torch.unique_consecutive(
                        cu_window_seqlens)
                    window_attention_mask = self.get_attention_mask(
                        cu_window_seqlens, seq_length)
                    reverse_window_index = torch.argsort(window_index)
                    inputs["window_attention_mask"] = window_attention_mask
                    inputs["window_index"] = window_index
                    inputs["reverse_window_index"] = reverse_window_index

                return inputs

        dataset = QwenViTDataset(dataset, model)
    else:
        raise NotImplementedError(f"Invalid model type {type(model)}")

    return dataset


def quantize_visual(model, precision, processor, dataset_dir="lmms-lab/MMMU"):
    assert isinstance(model,
                      Qwen2_5_VisionTransformerPretrainedModel) or isinstance(
                          model, Qwen2VisionTransformerPretrainedModel
                      ), f"Invalid model type {type(model)}"
    assert precision in [
        "fp8"
    ], f"Only fp8(W8A8) is supported for visual model. You passed an unsupported precision: {precision}."
    assert "MMMU" in dataset_dir, f"Unsupported dataset name or local repo directory: {dataset_dir}."

    # Set quantization config, this will only enable FP8 GEMMs that not belong to multihead attention modules.
    # Also disable Conv3d to avoid accuracy degradation.
    quant_config = mtq.FP8_DEFAULT_CFG.copy()
    quant_config["quant_cfg"]["nn.Conv3d"] = {"*": {"enable": False}}

    # With TensorRT 10.x, disable `attn.proj` layers to avoid performance degradation.
    quant_config["quant_cfg"]["*attn.proj*"] = {"enable": False}
    data_loader = get_visual_calib_dataloader(model, processor, dataset_dir)
    quantized_model = quantize_model(model, quant_config, data_loader)
    mtq.print_quant_summary(quantized_model)
    return quantized_model

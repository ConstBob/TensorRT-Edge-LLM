import os
import time

import modelopt.torch.quantization as mtq
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader


def get_calib_dataloader(dataset_name_or_dir="cnn_dailymail",
                         tokenizer=None,
                         batch_size=1,
                         calib_size=512,
                         block_size=512):
    print(f"Loading calibration dataset from {dataset_name_or_dir}")
    if "cnn_dailymail" in dataset_name_or_dir:
        dataset = load_dataset(dataset_name_or_dir,
                               name="3.0.0",
                               split="train")
        dataset = dataset["article"][:calib_size]
    elif os.path.isdir(dataset_name_or_dir):
        print(
            f"Recognized local dataset repo {dataset_name_or_dir} for calibration; "
            "assuming the calibration data are in the train split and text column."
        )
        dataset = load_dataset(dataset_name_or_dir, split="train")
        dataset = dataset["text"][:calib_size]
    else:
        raise NotImplementedError(
            f"Unsupported dataset name or local repo directory: {dataset_name_or_dir}."
        )

    batch_encoded = tokenizer.batch_encode_plus(dataset,
                                                return_tensors="pt",
                                                padding=True,
                                                truncation=True,
                                                max_length=block_size)

    calib_dataloader = DataLoader(batch_encoded["input_ids"],
                                  batch_size=batch_size,
                                  shuffle=False)

    return calib_dataloader


def get_quant_config(precision):

    if precision == "fp8":
        quant_cfg = mtq.FP8_DEFAULT_CFG
    elif precision == "int4":
        quant_cfg = mtq.INT4_AWQ_CFG
    return quant_cfg


def _quantize_model(model, precision, calib_dataloader=None):
    """
    The calibration loop for the model can be setup using the modelopt API.

    Example usage:
    from modelopt.torch.utils.dataset_utils import create_forward_loop
    model = ...  # Initilaize the model
    tokenizer = ...  # Initilaize the tokenizer
    quant_cfg = ...  # Setup quantization configuration
    forward_loop = create_forward_loop(model=model, dataset_name="cnn_dailymail", tokenizer=tokenizer)
    mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    """

    def calibrate_loop(model):
        """Adjusts weights and scaling factors based on selected algorithms."""
        for idx, data in enumerate(calib_dataloader):
            if idx % 10 == 0:
                print(f"Calibrating batch {idx}...")
            data = data.to(model.device)
            model(data)

    print("Starting quantization...")
    start_time = time.time()
    mtq.quantize(model,
                 get_quant_config(precision),
                 forward_loop=calibrate_loop)
    end_time = time.time()
    print(f"Quantization finishes in {end_time - start_time}s.")

    return model


def quantize(model, tokenizer, precision, dataset_dir=None):
    """
    Quantize the PyTorch model to fp8 or int4_awq
    """
    assert precision in [
        "fp8", "int4"
    ], f"Only fp8(W8A8) and int4(W4A16) is supported. You passed an unsupported precision: {precision}."
    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not dataset_dir:
        dataset_dir = "cnn_dailymail"

    if precision == "int4":
        batch_size = 32
    else:
        batch_size = 1
    data_loader = get_calib_dataloader(dataset_name_or_dir=dataset_dir,
                                       tokenizer=tokenizer,
                                       batch_size=batch_size)
    quantized_model = _quantize_model(model, precision, data_loader)
    mtq.print_quant_summary(quantized_model)
    return quantized_model

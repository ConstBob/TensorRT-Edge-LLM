"""
Test configuration utilities for TensorRT Edge-LLM tests.

This module provides a unified TestConfig class that handles both export and runtime 
test configurations, eliminating redundancy across the test framework.
"""

import enum
import os
from dataclasses import dataclass
from typing import Optional, Set

from conftest import EnvironmentConfig
from valid_precisions import (VALID_LLM_PRECISIONS, VALID_LM_HEAD_PRECISIONS,
                              VALID_VISUAL_PRECISIONS)


class ModelType(enum.Enum):
    """Supported model types"""
    LLM = "llm"
    VLM = "vlm"


class TaskType(enum.Enum):
    """Supported task types"""
    EXPORT = "export"
    BUILD = "build"
    BENCHMARK = "benchmark"
    INFERENCE = "inference"


@dataclass
class ParameterSpec:
    """Specification for a parameter with validation rules"""
    name: str
    format_hint: str
    required_for: Set[TaskType]
    valid_for: Set[ModelType]
    is_required: bool = True

    def is_valid_for_task_and_model(self, task_type: TaskType,
                                    model_type: ModelType) -> bool:
        """Check if this parameter is valid for the given task and model type"""
        return task_type in self.required_for and model_type in self.valid_for

    def is_required_for_task_and_model(self, task_type: TaskType,
                                       model_type: ModelType) -> bool:
        """Check if this parameter is required for the given task and model type"""
        return self.is_valid_for_task_and_model(
            task_type, model_type) and self.is_required


@dataclass
class TestConfig:
    """
    Test configuration class that handles both export and runtime test configurations.
    """
    # Core identifiers
    param_str: str
    model_name: str
    model_type: ModelType
    task_type: TaskType

    # Precision settings
    llm_precision: str
    lm_head_precision: Optional[str] = None
    visual_precision: Optional[str] = None

    # Directory paths
    torch_dir: Optional[str] = None
    onnx_dir: Optional[str] = None
    engine_dir: Optional[str] = None

    # Export LoRA parameters
    lora: Optional[bool] = None

    # Engine build parameters
    max_batch_size: Optional[int] = None
    max_input_len: Optional[int] = None
    max_seq_len: Optional[int] = None
    max_lora_rank: Optional[int] = None

    # VLM specific build parameters
    min_image_tokens: Optional[int] = None
    max_image_tokens: Optional[int] = None

    # Inference parameters
    test_case_file: Optional[str] = None

    # Benchmark parameters
    batch_size: Optional[int] = None
    input_seq_len: Optional[int] = None
    output_seq_len: Optional[int] = None

    # VLM-specific parameters (no defaults - must be specified)
    text_token_length: Optional[int] = None
    image_token_length: Optional[int] = None

    # Declarative parameter specifications
    _PARAMETER_SPECS = [
        # Core parameters for engine identification
        ParameterSpec(
            "max_seq_len", "mxsl", {
                TaskType.EXPORT, TaskType.BUILD, TaskType.BENCHMARK,
                TaskType.INFERENCE
            }, {ModelType.LLM, ModelType.VLM}),
        ParameterSpec("max_batch_size", "mxbs",
                      {TaskType.BUILD, TaskType.BENCHMARK, TaskType.INFERENCE},
                      {ModelType.LLM, ModelType.VLM}),
        ParameterSpec("max_input_len", "mxil",
                      {TaskType.BUILD, TaskType.BENCHMARK, TaskType.INFERENCE},
                      {ModelType.LLM, ModelType.VLM}),
        ParameterSpec("max_lora_rank",
                      "mxlr",
                      {TaskType.BUILD, TaskType.BENCHMARK, TaskType.INFERENCE},
                      {ModelType.LLM, ModelType.VLM},
                      is_required=False),

        # Export-specific parameters
        ParameterSpec("lora",
                      "", {TaskType.EXPORT}, {ModelType.LLM, ModelType.VLM},
                      is_required=False),

        # VLM-specific parameters
        ParameterSpec("min_image_tokens", "mnit",
                      {TaskType.BUILD, TaskType.BENCHMARK, TaskType.INFERENCE},
                      {ModelType.VLM}),
        ParameterSpec("max_image_tokens", "mxit",
                      {TaskType.BUILD, TaskType.BENCHMARK, TaskType.INFERENCE},
                      {ModelType.VLM}),
        ParameterSpec("visual_precision",
                      "vit", {
                          TaskType.EXPORT, TaskType.BUILD, TaskType.BENCHMARK,
                          TaskType.INFERENCE
                      }, {ModelType.VLM},
                      is_required=False),

        # Benchmark parameters
        ParameterSpec("batch_size", "bs", {TaskType.BENCHMARK},
                      {ModelType.LLM, ModelType.VLM}),
        ParameterSpec("input_seq_len", "isl", {TaskType.BENCHMARK},
                      {ModelType.LLM, ModelType.VLM}),
        ParameterSpec("output_seq_len", "osl", {TaskType.BENCHMARK},
                      {ModelType.LLM, ModelType.VLM}),
        ParameterSpec("text_token_length", "ttl", {TaskType.BENCHMARK},
                      {ModelType.VLM}),
        ParameterSpec("image_token_length", "itl", {TaskType.BENCHMARK},
                      {ModelType.VLM}),

        # Inference parameters
        ParameterSpec("test_case_file", "", {TaskType.INFERENCE},
                      {ModelType.LLM, ModelType.VLM}),
    ]

    @classmethod
    def from_param_string(cls, param_str: str, model_type: ModelType,
                          task_type: TaskType,
                          env_config: EnvironmentConfig) -> 'TestConfig':
        """
        Unified function to parse parameter string and create config with validation.
        
        Handles model names with multiple parts, performs all validation uniformly,
        and constructs the final config object.
        """

        # Validate environment based on task type
        if task_type == TaskType.EXPORT:
            env_config.validate_for_export_tests()
        else:
            env_config.validate_for_pipeline_tests()

        # Parse parameter string
        parts = param_str.split('-')

        # Find precision position and extract model name
        model_parts = []
        llm_precision = None
        lm_head_precision = None
        visual_precision = None
        remaining_parts = []

        # Find the first valid precision to determine where model name ends
        for i, part in enumerate(parts):
            if part in VALID_LLM_PRECISIONS:
                llm_precision = part
                model_parts = parts[:
                                    i]  # Everything before precision is model name

                # Check for lm head precision
                if i + 1 < len(parts) and parts[i + 1].startswith('lm'):
                    lm_head_precision = parts[i + 1][2:]
                    if lm_head_precision not in VALID_LM_HEAD_PRECISIONS:
                        raise ValueError(
                            f"Invalid LM head precision: {lm_head_precision}")
                    remaining_parts = parts[i + 2:]
                else:
                    remaining_parts = parts[i + 1:]
                break
        if not lm_head_precision:
            lm_head_precision = "fp16"

        if not llm_precision:
            raise ValueError(f"No valid precision found in: {param_str}")

        if not model_parts:
            raise ValueError(f"No model name found in: {param_str}")

        model_name = '-'.join(model_parts)

        # Parse remaining parameters
        parsed_params = {}

        for part in remaining_parts:
            # For engine identification
            if part.startswith('mxsl'):
                parsed_params['max_seq_len'] = int(part[4:])
            elif part == "lora":
                parsed_params['lora'] = True
            elif part.startswith('mxbs'):
                parsed_params['max_batch_size'] = int(part[4:])
            elif part.startswith('mxil'):
                parsed_params['max_input_len'] = int(part[4:])
            elif part.startswith('mnit'):
                parsed_params['min_image_tokens'] = int(part[4:])
            elif part.startswith('mxit'):
                parsed_params['max_image_tokens'] = int(part[4:])
            elif part.startswith('mxlr'):
                parsed_params['max_lora_rank'] = int(part[4:])
            # For benchmark parameters
            elif part.startswith('bs'):
                parsed_params['batch_size'] = int(part[2:])
            elif part.startswith('isl'):
                parsed_params['input_seq_len'] = int(part[3:])
            elif part.startswith('osl'):
                parsed_params['output_seq_len'] = int(part[3:])
            elif part.startswith('ttl'):
                parsed_params['text_token_length'] = int(part[3:])
            elif part.startswith('itl'):
                parsed_params['image_token_length'] = int(part[3:])
            elif part.startswith('vit'):
                visual_precision = part[3:]
                if visual_precision in VALID_VISUAL_PRECISIONS:
                    parsed_params['visual_precision'] = visual_precision
                else:
                    raise ValueError(
                        f"Invalid visual precision: {visual_precision}")
            # For inference parameters
            else:
                parsed_params['test_case_file'] = part

        if not visual_precision and model_type == ModelType.VLM:
            parsed_params['visual_precision'] = "fp16"

        # Create base config object
        config = cls(param_str=param_str,
                     model_name=model_name,
                     model_type=model_type,
                     task_type=task_type,
                     llm_precision=llm_precision,
                     lm_head_precision=lm_head_precision,
                     torch_dir=env_config.torch_dir
                     if task_type == TaskType.EXPORT else None,
                     onnx_dir=env_config.onnx_dir,
                     engine_dir=env_config.engine_dir
                     if task_type != TaskType.EXPORT else None)

        # Apply validated parameters to config
        for param_name, value in parsed_params.items():
            setattr(config, param_name, value)

        # Validate completeness and set defaults
        config._validate_completeness()

        return config

    def _validate_completeness(self) -> None:
        """Validate that all required parameters are set for the given task and model type"""

        def set_defaults() -> None:
            """Set default values for optional parameters"""
            if self.task_type == TaskType.EXPORT:
                if self.lora is None:
                    self.lora = False
            else:  # Runtime tasks
                if self.max_lora_rank is None:
                    self.max_lora_rank = 0
                if self.lora is None:
                    self.lora = self.max_lora_rank > 0

        missing_params = []
        invalid_params = []

        # Get valid and required parameters for current task/model combination
        valid_params = set()
        required_params = set()

        for spec in self._PARAMETER_SPECS:
            if spec.is_valid_for_task_and_model(self.task_type,
                                                self.model_type):
                valid_params.add(spec.name)
                if spec.is_required_for_task_and_model(self.task_type,
                                                       self.model_type):
                    required_params.add(spec.name)

        # Check for invalid parameters (parameters that are set but not allowed)
        for spec in self._PARAMETER_SPECS:
            param_value = getattr(self, spec.name)
            if param_value is not None and spec.name not in valid_params:
                invalid_params.append(spec.name)

        if invalid_params:
            task_desc = f"{self.model_type.value} {self.task_type.value}"
            raise ValueError(
                f"Invalid parameters for {task_desc}: {', '.join(invalid_params)}"
            )

        # Check for missing required parameters
        for spec in self._PARAMETER_SPECS:
            if spec.name in required_params and getattr(self,
                                                        spec.name) is None:
                param_desc = f'{spec.name} ({spec.format_hint})' if spec.format_hint else spec.name
                missing_params.append(param_desc)

        # Raise error if any required parameters are missing
        if missing_params:
            task_desc = f"{self.model_type.value} {self.task_type.value}"
            raise ValueError(
                f"Missing required parameters for {task_desc}: {', '.join(missing_params)}"
            )

        # Set defaults after validation
        set_defaults()

    # Unified path generation methods
    def get_onnx_model_id(self) -> str:
        """Generate unique model identifier"""
        return f"{self.llm_precision}-{self.lm_head_precision}-{self.max_seq_len}"

    def get_engine_id(self) -> str:
        """Generate unique engine identifier"""
        llm_engine_id = f"{self.get_onnx_model_id()}-mxil{self.max_input_len}-mxbs{self.max_batch_size}-mxlr{self.max_lora_rank}"
        if self.model_type == ModelType.VLM:
            llm_engine_id += f"-mnit{self.min_image_tokens}-mxit{self.max_image_tokens}"
        return llm_engine_id

    # TODO: Need a map-based approach to get the model directory instead of using the model name
    def get_torch_model_dir(self) -> str:
        """Get torch model directory path"""
        if not self.torch_dir:
            raise ValueError("torch_dir not set")
        return os.path.join(self.torch_dir, self.model_name)

    def get_onnx_base_dir(self) -> str:
        """Get ONNX model base directory"""
        if not self.onnx_dir:
            raise ValueError("onnx_dir not set")
        return os.path.join(self.onnx_dir, self.model_name)

    def get_engine_base_dir(self) -> str:
        """Get engine base directory"""
        if not self.engine_dir:
            raise ValueError("engine_dir not set")
        return os.path.join(self.engine_dir, self.model_name)

    def get_llm_onnx_dir(self) -> str:
        """Get LLM ONNX model directory"""
        return os.path.join(self.get_onnx_base_dir(),
                            f"llm-{self.get_onnx_model_id()}")

    def get_visual_onnx_dir(self, precision: str) -> str:
        """Get visual ONNX model directory"""
        return os.path.join(self.get_onnx_base_dir(), f"visual-{precision}")

    def get_llm_engine_dir(self) -> str:
        """Get LLM engine directory"""
        if not self.engine_dir:
            raise ValueError(
                "engine_dir not set - required for LLM engine operations")
        if self.task_type == TaskType.EXPORT:
            raise ValueError(
                "LLM engine directory not available for export tasks")
        return os.path.join(self.get_engine_base_dir(),
                            f"llm-{self.get_engine_id()}")

    def get_visual_engine_dir(self) -> str:
        """Get visual engine directory"""
        return os.path.join(
            self.get_engine_base_dir(),
            f"visual-{self.visual_precision}-mnit{self.min_image_tokens}-mxit{self.max_image_tokens}"
        )

    def get_test_case_file(self) -> str:
        """Get test case file path"""
        if not self.test_case_file:
            raise ValueError(
                "test_case_file not set - required for this operation")
        return f"tests/test_cases/{self.test_case_file}.json"

    def get_output_json_file(self) -> str:
        """Get output JSON file path"""
        if not self.engine_dir:
            raise ValueError("engine_dir not set")
        return os.path.join(self.get_engine_base_dir(),
                            f"{self.param_str}.json")

    def get_lora_weights_dir(self) -> str:
        """Get LoRA weights directory"""
        return os.path.join(self.get_onnx_base_dir(), "lora_weights")

    def get_quantized_model_dir(self) -> str:
        """Get quantized model directory (for export)"""
        if self.llm_precision == "fp16":
            return self.get_torch_model_dir()
        quantized_name = f"quantized-{self.llm_precision}-{self.lm_head_precision}-{self.max_seq_len}"
        return os.path.join(self.get_onnx_base_dir(), "quantized",
                            quantized_name)

    def get_cnn_dailymail_dataset_dir(self) -> str:
        """Get CNN DailyMail dataset directory for LLM quantization calibration"""
        return os.path.join(self.torch_dir, "datasets", "cnn_dailymail")

    def get_mmmu_dataset_dir(self) -> str:
        """Get MMMU dataset directory for visual model quantization calibration"""
        return os.path.join(self.torch_dir, "datasets", "MMMU")

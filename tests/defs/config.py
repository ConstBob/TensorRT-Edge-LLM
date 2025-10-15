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

    # EAGLE draft model precision settings
    draft_llm_precision: Optional[str] = None
    draft_lm_head_precision: Optional[str] = None

    # EAGLE model flag
    is_eagle: Optional[bool] = None

    # Directory paths
    llm_models_dir: Optional[str] = None
    edgellm_data_dir: Optional[str] = None
    onnx_dir: Optional[str] = None
    engine_dir: Optional[str] = None
    test_log_dir: Optional[str] = None

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
    max_image_tokens_per_image: Optional[int] = None

    # Inference parameters
    test_case: Optional[str] = None

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
        ParameterSpec("is_eagle",
                      "eagle", {TaskType.EXPORT},
                      {ModelType.LLM, ModelType.VLM},
                      is_required=False),
        ParameterSpec("draft_llm_precision",
                      "draft", {TaskType.EXPORT},
                      {ModelType.LLM, ModelType.VLM},
                      is_required=False),
        ParameterSpec("draft_lm_head_precision",
                      "draftlm", {TaskType.EXPORT},
                      {ModelType.LLM, ModelType.VLM},
                      is_required=False),

        # VLM-specific parameters
        ParameterSpec("min_image_tokens", "mnit",
                      {TaskType.BUILD, TaskType.BENCHMARK, TaskType.INFERENCE},
                      {ModelType.VLM}),
        ParameterSpec("max_image_tokens", "mxit",
                      {TaskType.BUILD, TaskType.BENCHMARK, TaskType.INFERENCE},
                      {ModelType.VLM}),
        ParameterSpec("max_image_tokens_per_image", "mxpiit",
                      {TaskType.BUILD, TaskType.BENCHMARK, TaskType.INFERENCE},
                      {ModelType.VLM}),
        ParameterSpec("visual_precision",
                      "vit", {
                          TaskType.EXPORT, TaskType.BUILD, TaskType.BENCHMARK,
                          TaskType.INFERENCE
                      }, {ModelType.VLM},
                      is_required=False),

        # Benchmark parameters
        ParameterSpec("input_seq_len", "isl", {TaskType.BENCHMARK},
                      {ModelType.LLM, ModelType.VLM}),
        ParameterSpec("output_seq_len", "osl", {TaskType.BENCHMARK},
                      {ModelType.LLM, ModelType.VLM}),
        ParameterSpec("text_token_length", "ttl", {TaskType.BENCHMARK},
                      {ModelType.VLM}),
        ParameterSpec("image_token_length", "itl", {TaskType.BENCHMARK},
                      {ModelType.VLM}),

        # Inference parameters
        ParameterSpec("test_case", "", {TaskType.INFERENCE},
                      {ModelType.LLM, ModelType.VLM}),
        ParameterSpec("batch_size",
                      "bs", {TaskType.INFERENCE},
                      {ModelType.LLM, ModelType.VLM},
                      is_required=False),
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
            elif part == "eagle":
                parsed_params['is_eagle'] = True
            elif part.startswith('draftlm'):
                draft_lm_precision = part[7:]
                if draft_lm_precision not in VALID_LM_HEAD_PRECISIONS:
                    raise ValueError(
                        f"Invalid draft LM head precision: {draft_lm_precision}"
                    )
                parsed_params['draft_lm_head_precision'] = draft_lm_precision
            elif part.startswith('draft'):
                draft_precision = part[5:]
                if draft_precision not in VALID_LLM_PRECISIONS:
                    raise ValueError(
                        f"Invalid draft precision: {draft_precision}")
                parsed_params['draft_llm_precision'] = draft_precision
            elif part.startswith('mxbs'):
                parsed_params['max_batch_size'] = int(part[4:])
            elif part.startswith('mxil'):
                parsed_params['max_input_len'] = int(part[4:])
            elif part.startswith('mnit'):
                parsed_params['min_image_tokens'] = int(part[4:])
            elif part.startswith('mxit'):
                parsed_params['max_image_tokens'] = int(part[4:])
            elif part.startswith('mxpiit'):
                parsed_params['max_image_tokens_per_image'] = int(part[6:])
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
                parsed_params['test_case'] = part

        if not visual_precision and model_type == ModelType.VLM:
            parsed_params['visual_precision'] = "fp16"

        # Create base config object
        config = cls(param_str=param_str,
                     model_name=model_name,
                     model_type=model_type,
                     task_type=task_type,
                     llm_precision=llm_precision,
                     lm_head_precision=lm_head_precision,
                     llm_models_dir=env_config.llm_models_dir
                     if task_type == TaskType.EXPORT else None,
                     edgellm_data_dir=env_config.edgellm_data_dir,
                     onnx_dir=env_config.onnx_dir,
                     engine_dir=env_config.engine_dir
                     if task_type != TaskType.EXPORT else None,
                     test_log_dir=env_config.test_log_dir)

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
                if self.is_eagle is None:
                    self.is_eagle = False
                # Set default draft model lm_head precision if draft model precision is set
                if self.draft_llm_precision is not None and self.draft_lm_head_precision is None:
                    self.draft_lm_head_precision = "fp16"
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

    def get_torch_model_dir(self) -> str:
        """
        Get torch model directory path using model name mapping.
        
        Raises:
            ValueError: If llm_models_dir is not set or model name is not supported
        """

        # Model name to base directory mapping
        # Maps logical model names to their actual directory names in llm_models_dir.
        # You need to adjust this map to match the actual directory names in your llm_models_dir.
        MODEL_NAME_TO_BASE_DIR_MAP = {
            "Qwen2.5-0.5B-Instruct":
            f"{self.llm_models_dir}/Qwen2.5-0.5B-Instruct",
            "Qwen2.5-1.5B-Instruct":
            f"{self.llm_models_dir}/Qwen2.5-1.5B-Instruct",
            "Qwen2.5-3B-Instruct":
            f"{self.llm_models_dir}/Qwen2.5-3B-Instruct",
            "Qwen2.5-7B-Instruct":
            f"{self.llm_models_dir}/Qwen2.5-7B-Instruct",
            "Qwen2.5-VL-3B-Instruct":
            f"{self.llm_models_dir}/Qwen2.5-VL-3B-Instruct",
            "Qwen2.5-VL-7B-Instruct":
            f"{self.llm_models_dir}/Qwen2.5-VL-7B-Instruct",
            "InternVL3-1B": f"{self.llm_models_dir}/InternVL3-1B-hf",
            "InternVL3-2B": f"{self.llm_models_dir}/InternVL3-2B-hf",
            "Llama-3.1-8B-Instruct":
            f"{self.llm_models_dir}/llama-3.1-model/Llama-3.1-8B-Instruct",
            "Llama-3.2-1B":
            f"{self.llm_models_dir}/llama-3.2-models/Llama-3.2-1B",
            "Llama-3.2-3B":
            f"{self.llm_models_dir}/llama-3.2-models/Llama-3.2-3B",
            "Qwen3-0.6B": f"{self.llm_models_dir}/Qwen3/Qwen3-0.6B",
            "Qwen3-8B": f"{self.llm_models_dir}/Qwen3/Qwen3-8B",
            # Add more mappings as needed
        }
        if self.model_name not in MODEL_NAME_TO_BASE_DIR_MAP:
            raise ValueError(
                f"Unsupported model name: '{self.model_name}'. "
                f"Supported models: {', '.join(MODEL_NAME_TO_BASE_DIR_MAP.keys())}"
            )
        model_dir_name = MODEL_NAME_TO_BASE_DIR_MAP[self.model_name]
        if not os.path.exists(model_dir_name):
            raise ValueError(f"Model directory not found: '{model_dir_name}'")
        return model_dir_name

    def get_draft_model_dir(self) -> str:
        """Get draft model directory"""
        # You need to adjust this map to match the actual directory names in your directories.
        # In our folder, the draft model is stored in the eagle_models folder.
        MODEL_NAME_TO_DRAFT_DIR_MAP = {
            "Qwen2.5-VL-7B-Instruct":
            f"{self.edgellm_data_dir}/eagle_models/qwen2.5-vl-7b-eagle3-v1",
            "Llama-3.1-8B-Instruct":
            f"{self.edgellm_data_dir}/eagle_models/EAGLE3-LLaMA3.1-Instruct-8B",
            "Qwen3-8B": f"{self.llm_models_dir}/Qwen3/qwen3_8b_eagle3",
            # Add more mappings as needed
        }
        if self.model_name not in MODEL_NAME_TO_DRAFT_DIR_MAP:
            raise ValueError(
                f"Unsupported model name: '{self.model_name}'. "
                f"Supported models: {', '.join(MODEL_NAME_TO_DRAFT_DIR_MAP.keys())}"
            )
        model_dir_name = MODEL_NAME_TO_DRAFT_DIR_MAP[self.model_name]
        if not os.path.exists(model_dir_name):
            raise ValueError(f"Model directory not found: '{model_dir_name}'")
        return model_dir_name

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

    def get_draft_onnx_model_id(self) -> str:
        """Generate unique draft model identifier"""
        if self.draft_llm_precision is None:
            raise ValueError("draft_llm_precision not set")
        if self.draft_lm_head_precision is None:
            raise ValueError("draft_lm_head_precision not set")
        return f"{self.draft_llm_precision}-{self.draft_lm_head_precision}-{self.max_seq_len}"

    def get_draft_onnx_dir(self) -> str:
        """Get draft model ONNX directory"""
        return os.path.join(self.get_onnx_base_dir(),
                            f"draft-{self.get_draft_onnx_model_id()}")

    def get_quantized_draft_model_dir(self) -> str:
        """Get quantized draft model directory (for export)"""
        if self.draft_llm_precision == "fp16":
            return self.get_draft_model_dir()
        quantized_name = f"quantized-{self.draft_llm_precision}-{self.draft_lm_head_precision}-{self.max_seq_len}"
        return os.path.join(self.get_onnx_base_dir(), "quantized-draft",
                            quantized_name)

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
            f"visual-{self.visual_precision}-mnit{self.min_image_tokens}-mxit{self.max_image_tokens}-mxpiit{self.max_image_tokens_per_image}"
        )

    def get_test_case_file(self) -> str:
        """
        Get test case file path using test case name mapping.
        
        Returns:
            Full path to the test case JSON file
            
        Raises:
            ValueError: If test_case is not set or not supported
        """
        if not self.test_case:
            raise ValueError("test_case not set - required for this operation")

        # Test case name to file path mapping
        # Maps logical test case names to their actual file paths.
        # You can adjust this map to match your test case organization.
        TEST_CASE_NAME_TO_PATH_MAP = {
            # Add test case mappings here, for example:
            "llm_basic":
            "tests/test_cases/llm_basic.json",
            "llm_lora":
            "tests/test_cases/llm_lora.json",
            "vlm_basic":
            "tests/test_cases/vlm_basic.json",
            "vlm_lora":
            "tests/test_cases/vlm_lora.json",
            "mtbench":
            f"{self.edgellm_data_dir}/datasets/MTBench/mtbench_eagle3.json",
            "mmmu":
            f"{self.edgellm_data_dir}/datasets/MMMU/mmmu_dataset.json",
            "mmmu_pro_4":
            f"{self.edgellm_data_dir}/datasets/MMMU_Pro_4/mmmu_pro_4_dataset.json",
            "mmmu_pro_10":
            f"{self.edgellm_data_dir}/datasets/MMMU_Pro_10/mmmu_pro_10_dataset.json",
            "mmmu_pro_vision":
            f"{self.edgellm_data_dir}/datasets/MMMU_Pro_vision/mmmu_pro_vision_dataset.json",
            "coco":
            f"{self.edgellm_data_dir}/datasets/coco/dataset.json",
            "mmlu_0":
            f"{self.edgellm_data_dir}/datasets/MMLU_zero_shot/mmlu_dataset.json",
            "mmlu_5":
            f"{self.edgellm_data_dir}/datasets/MMLU_five_shot/mmlu_dataset.json",
            "mmlu_pro":
            f"{self.edgellm_data_dir}/datasets/MMLU_Pro/mmlu_pro_dataset.json",
            "mmstar":
            f"{self.edgellm_data_dir}/datasets/MMStar/mmstar_reference.json",
            "aime":
            f"{self.edgellm_data_dir}/datasets/AIME/aime_dataset.json",
            "gsm8k":
            f"{self.edgellm_data_dir}/datasets/GSM8K/gsm8k_dataset.json",
            "gsm8k_10":
            f"{self.edgellm_data_dir}/datasets/GSM8K/gsm8k_dataset_10.json",
            "humaneval":
            f"{self.edgellm_data_dir}/datasets/HumanEval/humaneval_dataset.json",
            "math500":
            f"{self.edgellm_data_dir}/datasets/MATH500/math500_dataset.json",
        }

        if self.test_case not in TEST_CASE_NAME_TO_PATH_MAP:
            raise ValueError(
                f"Unsupported test case: '{self.test_case}'. "
                f"Supported test cases: {', '.join(TEST_CASE_NAME_TO_PATH_MAP.keys())}"
            )
        test_case_path = TEST_CASE_NAME_TO_PATH_MAP[self.test_case]
        if not os.path.exists(test_case_path):
            raise ValueError(f"Test case file not found: '{test_case_path}'")

        return test_case_path

    def get_output_json_file(self) -> str:
        """
        Get output JSON file path.
        Always stored on host in log directory for subsequent processing.
        """
        return os.path.join(self.test_log_dir, f"{self.param_str}.json")

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
        if not self.llm_models_dir:
            raise ValueError("llm_models_dir not set")
        return os.path.join(self.llm_models_dir, "datasets", "cnn_dailymail")

    def get_mmmu_dataset_dir(self) -> str:
        """Get MMMU dataset directory for visual model quantization calibration"""
        if not self.llm_models_dir:
            raise ValueError("llm_models_dir not set")
        return os.path.join(self.llm_models_dir, "datasets", "MMMU")

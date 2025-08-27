# Unified ONNX export configuration
# This file contains all ONNX export settings used across the codebase

# ONNX opset version for all exports
opset_version = 19

# Export settings
do_constant_folding = True
save_as_external_data = True
all_tensors_to_one_file = True
location = "onnx_model.data"
convert_attribute = True

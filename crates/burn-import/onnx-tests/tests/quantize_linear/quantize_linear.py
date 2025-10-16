#!/usr/bin/env python3
# file: quantize_linear.py

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

def create_model_without_zero_point():
    # Simple per-tensor quantization test
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [2, 2, 3, 4]
    )
    
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.INT8, [2, 2, 3, 4]
    )
    
    # Scale: scalar for per-tensor quantization
    scale_tensor = numpy_helper.from_array(
        np.array(0.1, dtype=np.float32), name="y_scale"
    )
    
    quantize_node = helper.make_node(
        "QuantizeLinear",
        inputs=["input", "y_scale"],
        outputs=["output"],
        name="quantize_linear"
    )
    
    graph_def = helper.make_graph(
        [quantize_node],
        "quantize_linear_test",
        [input_tensor],
        [output_tensor],
        [scale_tensor]
    )
    
    model_def = helper.make_model(
        graph_def,
        producer_name="quantize_linear_test",
        opset_imports=[helper.make_opsetid("", 19)]
    )
    
    onnx.save(model_def, "quantize_linear.onnx")
    print("Model saved as quantize_linear.onnx")
    return model_def

def create_model_with_zero_point():
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [2, 2, 3, 4]
    )
    
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.INT8, [2, 2, 3, 4]
    )
    
    scale_tensor = numpy_helper.from_array(
        np.array(0.1, dtype=np.float32), name="y_scale"
    )
    
    zero_point_tensor = numpy_helper.from_array(
        np.array(10, dtype=np.int8), name="y_zero_point"
    )
    
    quantize_node = helper.make_node(
        "QuantizeLinear",
        inputs=["input", "y_scale", "y_zero_point"],
        outputs=["output"],
        name="quantize_linear"
    )
    
    graph_def = helper.make_graph(
        [quantize_node],
        "quantize_linear_test",
        [input_tensor],
        [output_tensor],
        [scale_tensor, zero_point_tensor]
    )
    
    model_def = helper.make_model(
        graph_def,
        producer_name="quantize_linear_test",
        opset_imports=[helper.make_opsetid("", 19)]
    )
    
    onnx.save(model_def, "quantize_linear_with_zero_point.onnx")
    print("Model saved as quantize_linear_with_zero_point.onnx")
    return model_def

if __name__ == "__main__":
    create_model_without_zero_point()
    create_model_with_zero_point()
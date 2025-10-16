#!/usr/bin/env python3
# file: quantize_linear.py

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def create_model_without_zero_point():
    # Simple per-tensor quantization test
    # Input: float32 tensor
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [2, 3, 4]
    )
    
    # Output: quantized int8 tensor
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.INT8, [2, 3, 4]
    )
    
    # Scale: scalar for per-tensor quantization
    scale_tensor = numpy_helper.from_array(
        np.array(0.1, dtype=np.float32), name="y_scale"
    )
    
    # Create QuantizeLinear node
    quantize_node = helper.make_node(
        "QuantizeLinear",
        inputs=["input", "y_scale"],
        outputs=["output"],
        name="quantize_linear"
    )
    
    # Create the graph
    graph_def = helper.make_graph(
        [quantize_node],
        "quantize_linear_test",
        [input_tensor],
        [output_tensor],
        [scale_tensor]
    )
    
    # Create the model with opset 16
    model_def = helper.make_model(
        graph_def,
        producer_name="quantize_linear_test",
        opset_imports=[helper.make_opsetid("", 16)]
    )
    
    # Save the model
    onnx.save(model_def, "quantize_linear.onnx")
    print("Model saved as quantize_linear.onnx")

    return model_def


def create_model_with_zero_point():
    # Simple per-tensor quantization test
    # Input: float32 tensor
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [2, 3, 4]
    )
    
    # Output: quantized int8 tensor
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.INT8, [2, 3, 4]
    )
    
    # Scale: scalar for per-tensor quantization
    scale_tensor = numpy_helper.from_array(
        np.array(0.1, dtype=np.float32), name="y_scale"
    )
    
    # Zero point: scalar int8
    zero_point_tensor = numpy_helper.from_array(
        np.array(0, dtype=np.int8), name="y_zero_point"
    )
    
    # Create QuantizeLinear node
    quantize_node = helper.make_node(
        "QuantizeLinear",
        inputs=["input", "y_scale", "y_zero_point"],
        outputs=["output"],
        name="quantize_linear"
    )
    
    # Create the graph
    graph_def = helper.make_graph(
        [quantize_node],
        "quantize_linear_test",
        [input_tensor],
        [output_tensor],
        [scale_tensor, zero_point_tensor]
    )
    
    # Create the model with opset 16
    model_def = helper.make_model(
        graph_def,
        producer_name="quantize_linear_test",
        opset_imports=[helper.make_opsetid("", 16)]
    )
    
    # Save the model
    onnx.save(model_def, "quantize_linear.onnx")
    print("Model saved as quantize_linear.onnx")

    return model_def

def main():
    # Test with sample data
    try:
        model_def = create_model_without_zero_point()

    except Exception as e:
        print(f"Error: {e}")
    try:
        from onnx.reference import ReferenceEvaluator
        
        test_input = np.random.randn(2, 3, 4).astype(np.float32)
        print(f"\nTest input shape: {test_input.shape}")
        print(f"Test input sample: {test_input[0, 0, :]}")
        
        session = ReferenceEvaluator(model_def, verbose=0)
        output, = session.run(None, {"input": test_input})
        
        print(f"Test output shape: {output.shape}")
        print(f"Test output sample: {output[0, 0, :]}")
        print(f"Test output dtype: {output.dtype}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
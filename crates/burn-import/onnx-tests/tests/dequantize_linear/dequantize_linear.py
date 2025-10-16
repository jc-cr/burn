#!/usr/bin/env python3
# file: dequantize_linear.py
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator
import traceback

def create_model_without_zero_point():
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.INT8, [2, 3, 4]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [2, 3, 4]
    )
    
    # Scale: scalar for per-tensor dequantization
    scale_tensor = numpy_helper.from_array(
        np.array(0.1, dtype=np.float32), name="x_scale"
    )
    
    # Create DequantizeLinear node
    dequantize_node = helper.make_node(
        "DequantizeLinear",
        inputs=["input", "x_scale"],
        outputs=["output"],
        name="dequantize_linear"
    )
    
    # Create the graph
    graph_def = helper.make_graph(
        [dequantize_node],
        "dequantize_linear_test",
        [input_tensor],
        [output_tensor],
        [scale_tensor]
    )
    
    model_def = helper.make_model(
        graph_def,
        producer_name="dequantize_linear_test",
        opset_imports=[helper.make_opsetid("", 19)]
    )
    
    onnx.save(model_def, "dequantize_linear.onnx")
    print("Model saved as dequantize_linear.onnx")
    return model_def

def create_model_with_zero_point():
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.INT8, [2, 3, 4]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [2, 3, 4]
    )
    
    # Scale: scalar for per-tensor dequantization
    scale_tensor = numpy_helper.from_array(
        np.array(0.1, dtype=np.float32), name="x_scale"
    )
    
    # Zero point: scalar int8 (non-zero for testing)
    zero_point_tensor = numpy_helper.from_array(
        np.array(-95, dtype=np.int8), name="x_zero_point"
    )
    
    # Create DequantizeLinear node
    dequantize_node = helper.make_node(
        "DequantizeLinear",
        inputs=["input", "x_scale", "x_zero_point"],
        outputs=["output"],
        name="dequantize_linear"
    )
    
    # Create the graph
    graph_def = helper.make_graph(
        [dequantize_node],
        "dequantize_linear_test",
        [input_tensor],
        [output_tensor],
        [scale_tensor, zero_point_tensor]
    )
    
    model_def = helper.make_model(
        graph_def,
        producer_name="dequantize_linear_test",
        opset_imports=[helper.make_opsetid("", 19)]
    )
    
    onnx.save(model_def, "dequantize_linear_with_zero_point.onnx")  # Different filename!
    print("Model saved as dequantize_linear_with_zero_point.onnx")
    return model_def

def test_model_without_zero_point():
    print("\n=== Testing WITHOUT zero point ===")
    try:
        model_def = create_model_without_zero_point()
        
        test_input = np.random.randint(-128, 128, size=(2, 3, 4), dtype=np.int8)
        print(f"Test input shape: {test_input.shape}")
        print(f"Test input sample: {test_input[0, 0, :]}")
        
        session = ReferenceEvaluator(model_def, verbose=0)
        output, = session.run(None, {"input": test_input})
        
        print(f"Test output shape: {output.shape}")
        print(f"Test output sample: {output[0, 0, :]}")
        print(f"Test output dtype: {output.dtype}")
        
        # Verify: y = x * scale
        expected = test_input[0, 0, :] * 0.1
        print(f"Expected (x * scale): {expected}")
        print(f"Match: {np.allclose(output[0, 0, :], expected)}")

    except Exception as e:
        raise e


def test_model_with_zero_point():
    print("\n=== Testing WITH zero point ===")
    try:
        model_def = create_model_with_zero_point()
        
        test_input = np.random.randint(-128, 128, size=(2, 3, 4), dtype=np.int8)
        print(f"Test input shape: {test_input.shape}")
        print(f"Test input sample: {test_input[0, 0, :]}")
        
        session = ReferenceEvaluator(model_def, verbose=0)
        output, = session.run(None, {"input": test_input})
        
        print(f"Test output shape: {output.shape}")
        print(f"Test output sample: {output[0, 0, :]}")
        print(f"Test output dtype: {output.dtype}")
        
        # Verify: y = (x - zero_point) * scale
        zero_point = -95
        scale = 0.1
        expected = (test_input[0, 0, :] - zero_point) * scale
        print(f"Expected ((x - {zero_point}) * {scale}): {expected}")
        print(f"Match: {np.allclose(output[0, 0, :], expected)}")

    except Exception as e:
        raise e
            

if __name__ == "__main__":
    try:
        test_model_without_zero_point()
    except Exception as e:
        print(f"Error in test without zero point: {e}")
        traceback.print_exc()

    try:
        test_model_with_zero_point()
    except Exception as e:
        print(f"Error in test with zero point: {e}")
        traceback.print_exc()
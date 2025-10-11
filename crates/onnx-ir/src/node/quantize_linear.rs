use crate::ir::*;

pub fn quantize_linear_config(node: &Node) -> (Option<i64>, Option<i64>) {
    // Extract axis attribute if present (ONNX opset 13+)
    let axis = node.attrs.get("axis").and_then(|v| v.clone().into_i64());
    
    // Check if zero_point is provided (3rd input)
    let has_zero_point = node.inputs.len() > 2;
    
    (axis, if has_zero_point { Some(1) } else { None })
}

pub fn quantize_linear_update_output(node: &mut Node) {
    // QuantizeLinear maintains the same shape, but changes element type to Int8
    if let ArgType::Tensor(input_tensor) = &node.inputs[0].ty {
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: ElementType::Int8, // Quantized output
            rank: input_tensor.rank,
            static_shape: input_tensor.static_shape.clone(),
        });
    }
}
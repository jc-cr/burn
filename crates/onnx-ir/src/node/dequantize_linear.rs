use crate::ir::*;

pub fn dequantize_linear_config(node: &Node) -> (Option<i64>, Option<i64>) {
    // Extract axis attribute if present
    let axis = node.attrs.get("axis").map(|v| v.clone().into_i64());
    
    // Check if zero_point is provided (3rd input)
    let has_zero_point = node.inputs.len() > 2;
    
    (axis, if has_zero_point { Some(1) } else { None })
}

pub fn dequantize_linear_update_output(node: &mut Node) {
    // DequantizeLinear maintains the same shape, but changes element type to Float32
    if let ArgType::Tensor(input_tensor) = &node.inputs[0].ty {
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: ElementType::Float32, // Dequantized output (float)
            rank: input_tensor.rank,
            static_shape: input_tensor.static_shape.clone(),
        });
    }
}
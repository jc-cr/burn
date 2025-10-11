use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone)]
pub struct QuantizeLinearNode {
    pub input: TensorType,
    pub scale: TensorType,
    pub zero_point: Option<TensorType>,
    pub output: TensorType,
    pub axis: Option<i64>,
}

impl QuantizeLinearNode {
    pub fn new(
        input: TensorType,
        scale: TensorType,
        zero_point: Option<TensorType>,
        output: TensorType,
        axis: Option<i64>,
    ) -> Self {
        Self {
            input,
            scale,
            zero_point,
            output,
            axis,
        }
    }
}

impl OnnxIntoNode for QuantizeLinearNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let input = TensorType::from(node.inputs.first().unwrap());
        let scale = TensorType::from(node.inputs.get(1).unwrap());
        let zero_point = node.inputs.get(2).map(TensorType::from);
        let output = TensorType::from(node.outputs.first().unwrap());
        
        let (axis, _) = onnx_ir::node::quantize_linear::quantize_linear_config(&node);

        Self::new(input, scale, zero_point, output, axis)
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for QuantizeLinearNode {
    fn input_types(&self) -> Vec<Type> {
        let mut types = vec![Type::Tensor(self.input.clone()), Type::Tensor(self.scale.clone())];
        if let Some(ref zp) = self.zero_point {
            types.push(Type::Tensor(zp.clone()));
        }
        types
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let scale = scope.tensor_use_owned(&self.scale, node_position);
        let output = &self.output.name();

        // For now, basic implementation
        // You'll need to check Burn's API for actual quantization operations
        if let Some(ref zp) = self.zero_point {
            let zero_point = scope.tensor_use_owned(zp, node_position);
            quote! {
                let #output = #input.quantize_linear(#scale, Some(#zero_point));
            }
        } else {
            quote! {
                let #output = #input.quantize_linear(#scale, None);
            }
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::QuantizeLinear(self)
    }
}

#[cfg(test)]
mod tests {
    // Add tests here
}
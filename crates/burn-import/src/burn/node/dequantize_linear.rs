use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone)]
pub struct DequantizeLinearNode {
    pub input: Type,
    pub scale: Type,
    pub zero_point: Option<Type>,
    pub output: Type,
    pub axis: Option<i64>,
}

impl DequantizeLinearNode {
    pub fn new(
        input: Type,
        scale: Type,
        zero_point: Option<Type>,
        output: Type,
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

impl OnnxIntoNode for DequantizeLinearNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let input = Type::from(node.inputs.first().unwrap());
        let scale = Type::from(node.inputs.get(1).unwrap());
        let zero_point = node.inputs.get(2).map(Type::from);
        let output = Type::from(node.outputs.first().unwrap());
        
        let (axis, _) = onnx_ir::node::dequantize_linear::dequantize_linear_config(&node);

        Self::new(input, scale, zero_point, output, axis)
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for DequantizeLinearNode {
    fn input_types(&self) -> Vec<Type> {
        let mut types = vec![self.input.clone(), self.scale.clone()];
        if let Some(ref zp) = self.zero_point {
            types.push(zp.clone());
        }
        types
    }

    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(self.input.as_tensor(), node_position);
        let scale_name = self.scale.name();
        let output = self.output.name();

        // ONNX DequantizeLinear formula: y = (x - x_zero_point) * x_scale
        // Without zero_point: y = x * x_scale
        
        if self.zero_point.is_some() {
            // TODO: Implement with zero_point
            quote! {
                compile_error!("DequantizeLinear with zero_point is not yet implemented");
            }
        } else {
            // Formula: y = x.float() * scale
            match self.axis {
                None => {
                    // Per-tensor dequantization (scale is scalar constant)
                    quote! {
                        // Dequantize: convert to float and multiply by scale
                        let #output = #input
                            .float()
                            .mul_scalar(#scale_name);
                    }
                }
                Some(_axis) => {
                    // Per-channel dequantization
                    quote! {
                        compile_error!("DequantizeLinear with axis (per-channel) is not yet implemented");
                    }
                }
            }
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::DequantizeLinear(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{graph::BurnGraph, node::test::assert_tokens, ScalarKind, ScalarType, TensorType};
    use burn::record::FullPrecisionSettings;

    #[test]
    fn test_codegen_dequantize_linear_without_zero_point() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(DequantizeLinearNode::new(
            Type::Tensor(TensorType::new_int("input", 4)),
            Type::Scalar(ScalarType::new("scale", ScalarKind::Float32)),
            None, // No zero point
            Type::Tensor(TensorType::new_float("output", 4)),
            None, // Per-tensor (no axis)
        ));

        graph.register_input_output(vec!["input".to_string()], vec!["output".to_string()]);

        let expected = quote! {
            use burn::tensor::Int;
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }

                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self, input: Tensor<B, 4, Int>) -> Tensor<B, 4> {
                    let scale = 0.1f32;

                    let output = input
                        .float()
                        .mul_scalar(scale);

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
use crate::include_models;

include_models!(
    dequantize_linear,
    dequantize_linear_with_zero_point
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;
    use crate::backend::TestBackend;

    #[test]
    fn dequantize_linear_without_zero_point() {
        let device = Default::default();
        let model = dequantize_linear::Model::<TestBackend>::new(&device);

        // Test input - int8 quantized values
        let input = Tensor::<TestBackend, 4, burn::tensor::Int>::from_ints(
            [
                [
                    [
                        [5, 10, 15, 20],
                        [25, 30, 35, 40],
                        [45, 50, 55, 60],
                    ],
                    [
                        [65, 70, 75, 80],
                        [85, 90, 95, 100],
                        [105, 110, 115, 120],
                    ],
                ],
                [
                    [
                        [-5, -10, -15, -20],
                        [-25, -30, -35, -40],
                        [-45, -50, -55, -60],
                    ],
                    [
                        [-65, -70, -75, -80],
                        [-85, -90, -95, -100],
                        [-105, -110, -115, -120],
                    ],
                ],
            ],
            &device,
        );

        let output = model.forward(input);
        
        // With scale = 0.1, y = x * 0.1
        let expected = Tensor::<TestBackend, 4>::from_floats(
            [
                [
                    [
                        [0.5, 1.0, 1.5, 2.0],
                        [2.5, 3.0, 3.5, 4.0],
                        [4.5, 5.0, 5.5, 6.0],
                    ],
                    [
                        [6.5, 7.0, 7.5, 8.0],
                        [8.5, 9.0, 9.5, 10.0],
                        [10.5, 11.0, 11.5, 12.0],
                    ],
                ],
                [
                    [
                        [-0.5, -1.0, -1.5, -2.0],
                        [-2.5, -3.0, -3.5, -4.0],
                        [-4.5, -5.0, -5.5, -6.0],
                    ],
                    [
                        [-6.5, -7.0, -7.5, -8.0],
                        [-8.5, -9.0, -9.5, -10.0],
                        [-10.5, -11.0, -11.5, -12.0],
                    ],
                ],
            ],
            &device,
        );

        output.to_data().assert_approx_eq(&expected.to_data(), 3);
    }

    #[test]
    fn dequantize_linear_with_zero_point() {
        let device = Default::default();
        let model = dequantize_linear_with_zero_point::Model::<TestBackend>::new(&device);

        // Test input - int8 quantized values
        let input = Tensor::<TestBackend, 4, burn::tensor::Int>::from_ints(
            [
                [
                    [
                        [10, 5, 0, -5],
                        [-10, -15, -20, -25],
                        [-30, -35, -40, -45],
                    ],
                    [
                        [-50, -55, -60, -65],
                        [-70, -75, -80, -85],
                        [-90, -95, -100, -105],
                    ],
                ],
                [
                    [
                        [15, 20, 25, 30],
                        [35, 40, 45, 50],
                        [55, 60, 65, 70],
                    ],
                    [
                        [75, 80, 85, 90],
                        [95, 100, 105, 110],
                        [115, 120, 125, 127],
                    ],
                ],
            ],
            &device,
        );

        let output = model.forward(input);
        
        // With zero_point = -95 and scale = 0.1
        // y = (x - (-95)) * 0.1 = (x + 95) * 0.1
        let expected = Tensor::<TestBackend, 4>::from_floats(
            [
                [
                    [
                        [10.5, 10.0, 9.5, 9.0],
                        [8.5, 8.0, 7.5, 7.0],
                        [6.5, 6.0, 5.5, 5.0],
                    ],
                    [
                        [4.5, 4.0, 3.5, 3.0],
                        [2.5, 2.0, 1.5, 1.0],
                        [0.5, 0.0, -0.5, -1.0],
                    ],
                ],
                [
                    [
                        [11.0, 11.5, 12.0, 12.5],
                        [13.0, 13.5, 14.0, 14.5],
                        [15.0, 15.5, 16.0, 16.5],
                    ],
                    [
                        [17.0, 17.5, 18.0, 18.5],
                        [19.0, 19.5, 20.0, 20.5],
                        [21.0, 21.5, 22.0, 22.2],
                    ],
                ],
            ],
            &device,
        );

        output.to_data().assert_approx_eq(&expected.to_data(), 3);
    }
}
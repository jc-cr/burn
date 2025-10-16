use crate::include_models;

include_models!(
    quantize_linear,
    quantize_linear_with_zero_point
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;
    use crate::backend::TestBackend;

    #[test]
    fn quantize_linear_without_zero_point() {
        let device = Default::default();
        let model = quantize_linear::Model::<TestBackend>::new(&device);

        // Test input
        let input = Tensor::<TestBackend, 4>::from_floats(
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

        let output = model.forward(input);
        
        let expected = Tensor::<TestBackend, 4, burn::tensor::Int>::from_ints(
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

        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn quantize_linear_with_zero_point() {
        let device = Default::default();
        let model = quantize_linear_with_zero_point::Model::<TestBackend>::new(&device);

        // Test input
        let input = Tensor::<TestBackend, 4>::from_floats(
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

        let output = model.forward(input);
        
        // With zero_point = 10 and scale = 0.1
        // y = round(x / 0.1) + 10 = round(x * 10) + 10
        let expected = Tensor::<TestBackend, 4, burn::tensor::Int>::from_ints(
            [
                [
                    [
                        [15, 20, 25, 30],
                        [35, 40, 45, 50],
                        [55, 60, 65, 70],
                    ],
                    [
                        [75, 80, 85, 90],
                        [95, 100, 105, 110],
                        [115, 120, 125, 127], // 127 is clamped max
                    ],
                ],
                [
                    [
                        [5, 0, -5, -10],
                        [-15, -20, -25, -30],
                        [-35, -40, -45, -50],
                    ],
                    [
                        [-55, -60, -65, -70],
                        [-75, -80, -85, -90],
                        [-95, -100, -105, -110],
                    ],
                ],
            ],
            &device,
        );

        output.to_data().assert_eq(&expected.to_data(), true);
    }
}
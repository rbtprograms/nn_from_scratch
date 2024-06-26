use crate::layer::Layer;
use std::ops::Neg;
use std::time::Instant;

pub struct NN {
    pub layers: Vec<Layer>,
}

impl NN {
    pub fn new() -> NN {
        NN { layers: Vec::new() }
    }
    pub fn add_layer(&mut self, input_size: usize, output_size: usize) {
        let layer = Layer::new(input_size, output_size);
        self.layers.push(layer)
    }

    pub fn train(&mut self, data: &[(Vec<f64>, Vec<f64>)], epochs: usize, learning_rate: f64) {
        for i in 0..epochs {
            let now = Instant::now();
            println!["Epoch {} begin", i + 1];
            let mut epoch_loss = 0.0;

            for (input, target) in data {
                // Implement forward pass, loss calculation, backpropagation, and weight updates
                let mut pre_activations_list = Vec::new();
                let mut post_activations_list = Vec::new();
                post_activations_list.push(input.clone());

                for layer in &mut self.layers {
                    let curr_inputs = post_activations_list.last().unwrap();
                    let (pre_activations, post_activations) = layer.forward(curr_inputs);
                    pre_activations_list.push(pre_activations);
                    post_activations_list.push(post_activations);
                }

                let final_output = post_activations_list.last().unwrap();
                let mut output_grad = cross_entropy_loss_prime(target, final_output);
                epoch_loss += cross_entropy_loss(target, final_output);

                for (i, layer) in self.layers.iter_mut().enumerate().rev() {
                    let input = &post_activations_list[i];
                    let pre_activations = &pre_activations_list[i];
                    output_grad = layer.backward(input, pre_activations, &output_grad);
                }

                for layer in &mut self.layers {
                    layer.update_weights(learning_rate);
                }
            }
            println![
                "Epoch {} complete, took {:?}, Loss: {}",
                i + 1,
                now.elapsed(),
                epoch_loss / data.len() as f64
            ];
        }
    }

    pub fn predict(&mut self, input: &[f64]) -> Vec<f64> {
        let mut activation = input.to_vec();
        for layer in &mut self.layers {
            // println!("Weights at prediction: {:?}", layer.print_weights());
            let (_, post_activation) = layer.forward(&activation);
            activation = post_activation;
        }
        activation
    }
}

pub fn cross_entropy_loss(y_true: &Vec<f64>, y_hat: &Vec<f64>) -> f64 {
    y_true
        .iter()
        .zip(y_hat.iter())
        .map(|(y, y_hat)| y * y_hat.ln())
        .sum::<f64>()
        .neg()
}

pub fn cross_entropy_loss_prime(y_true: &[f64], y_hat: &[f64]) -> Vec<f64> {
    y_true
        .iter()
        .zip(y_hat.iter())
        .map(|(y, y_hat)| y_hat - y) // This is correct for cross-entropy with sigmoid output
        .collect()
}

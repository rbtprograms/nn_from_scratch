use rand::Rng;
use std::ops::Neg;
use std::time::Instant;

pub struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    // NOTE: add grad part later in article
    weight_grad: Vec<Vec<f64>>,
    bias_grad: Vec<f64>,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize) -> Layer {
        let mut rng = rand::thread_rng();
        let weights = (0..output_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        let weight_grad = vec![vec![0.0; input_size]; output_size];
        let biases = vec![0.0; output_size];
        let bias_grad = vec![0.0; output_size];
        // println!("intialized Weights: {:?}", weights);
        Layer {
            weights,
            weight_grad,
            biases,
            bias_grad,
        }
    }

    pub fn forward(&mut self, input: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let output_size = self.biases.len();
        let mut pre_activations = vec![0.0; output_size];
        let mut post_activations = vec![0.0; output_size];

        // comput a dot product of input against each weight vector
        for (i, weight) in self.weights.iter().enumerate() {
            let mut dot = 0.0;
            for (j, weight_) in weight.iter().enumerate() {
                dot += weight_ * input[j];
            }
            pre_activations[i] = dot + self.biases[i];
            //want to map each weighted sum to the sigmoid activation
            post_activations[i] = sigmoid(pre_activations[i]);
        }
        (pre_activations, post_activations)
    }

    // backward needs to return the error for this layer
    fn backward(
        &mut self,
        input: &Vec<f64>,
        pre_activations: &Vec<f64>,
        output_grad: &Vec<f64>,
    ) -> Vec<f64> {
        let input_size = input.len();
        let output_size = self.biases.len();
        let mut input_grad = vec![0.0; input_size];

        // want to find dE/dW for each weight, which is given by activation*output_error
        for i in 0..output_size {
            let delta = output_grad[i] * sigmoid_prime(pre_activations[i]);
            for j in 0..input_size {
                // updates each weight gradient, which we save on the struct
                self.weight_grad[i][j] = input[j] * delta;
                // summation for the error for each neuron in the layer
                input_grad[j] += delta * self.weights[i][j];
            }
            // bias gradient is just the output error
            self.bias_grad[i] = delta;
        }

        input_grad
    }

    pub fn print_weights(&mut self) {
        println!("{:?}", self.weights);
    }

    fn update_weights(&mut self, learning_rate: f64) {
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                self.weights[i][j] -= self.weight_grad[i][j] * learning_rate;
                self.weight_grad[i][j] = 0.0; // Reset gradient after update
            }
            self.biases[i] -= self.bias_grad[i] * learning_rate;
            self.bias_grad[i] = 0.0; // Reset gradient after update
        }
    }
}

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
            println!["Epoch {} begin", i+1];
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

                for (i, layer) in self.layers.iter_mut().enumerate().rev() {
                    let input = &post_activations_list[i];
                    let pre_activations = &pre_activations_list[i];
                    output_grad = layer.backward(input, pre_activations, &output_grad);
                }

                for layer in &mut self.layers {
                    layer.update_weights(learning_rate);
                }
            }
            println!["Epoch {} complete, took {:?}", i+1, now.elapsed()];
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

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_prime(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
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

// fn softmax(logits: &Vec<f64>) -> Vec<f64> {
//     //we arent dealing with overflow rn, will cross bridge if we get there
//     let logit_exp_sum = logits.iter().map(|x| x.exp()).sum::<f64>();
//     logits
//         .iter()
//         .map(|logit| logit.exp() / logit_exp_sum)
//         .collect()
// }

use std::ops::Neg;

struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    // NOTE: add grad part later in article
    weight_grad: Vec<Vec<f64>>,
    bias_grad: Vec<f64>,
}

impl Layer {
    pub fn new(input_size:usize, output_size:usize) -> Layer {
        let weights = vec![vec![0.0; input_size]; output_size];
        let weight_grad = vec![vec![0.0; input_size]; output_size];
        let biases = vec![0.0; output_size];
        let bias_grad = vec![0.0; output_size];
        Layer { weights, weight_grad, biases, bias_grad }
    }

    fn forward(&mut self, input: &Vec<f64>) -> Vec<f64> {
        let output_size = self.biases.len();
        let mut output = vec![0.0; output_size];

        // comput a dot product of input against each weight vector
        for (i, weight) in self.weights.iter().enumerate() {
            let mut dot = 0.0;
            for (j, weight_) in weight.iter().enumerate() {
                dot += weight_ * input[j];
            }
            output[i] = dot + self.biases[i];
        }

        // map each output to a sigmoid
        output.iter().map(|&x| sigmoid(x)).collect()
    }

    // backward needs to return the error for this layer
    fn backward(&mut self, input: &Vec<f64>, output_grad: &Vec<f64>) -> Vec<f64> {
        let input_size = input.len();
        let output_size = self.biases.len();

        let mut output = vec![0.0; output_size];

        // want to find dE/dW for each weight, which is given by activation*output_error
        for i in 0..output_size {
            for j in 0..input_size {
                // updates each weight gradient, which we save on the struct
                self.weight_grad[i][j] = input[j] * output_grad[i];
                // summation for the error for each neuron in the layer
                output[j] += output_grad[i] * self.weights[i][j];
            }
            // bias gradient is just the output error
            self.bias_grad[i] = output_grad[i];
        }

        output
    }

    fn update_weights(&mut self, learning_rate: f64) {
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                self.weights[i][j] -= self.weight_grad[i][j] * learning_rate
            }
            self.biases[i] -= self.bias_grad[i] * learning_rate;
        }
    }
}

struct NN {
    layers: Vec<Layer>,
    // data: (Vec<f64>, Vec<f64>),
    learning_rate: f64,
    epochs: usize
}

impl NN {
    pub fn new(learning_rate: f64, epochs: usize) -> NN {
        NN { 
            learning_rate,
            epochs,
            layers: Vec::new()
        } 
    }
    pub fn add_layer(&mut self, input_size: usize, output_size: usize) {
        let layer = Layer::new(input_size, output_size);
        self.layers.push(layer)
    }

    fn train(&mut self, data: &[(Vec<f64>, Vec<f64>)], epochs: usize, learning_rate: f64) {
        for _ in 0..epochs {
            for (input, target) in data {
                // Implement forward pass, loss calculation, backpropagation, and weight updates
            }
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}


fn relu(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        x
    }
}

fn cross_entropy_loss(y_true: &Vec<f64>, y_hat: &Vec<f64>) -> f64 {
    y_true
        .iter()
        .zip(y_hat.iter())
        .map(|(y, y_hat)| y * y_hat.ln())
        .sum::<f64>()
        .neg()
}

fn softmax(logits: &Vec<f64>) -> Vec<f64> {
    //we arent dealing with overflow rn, will cross bridge if we get there
    let logit_exp_sum = logits.iter().map(|x| x.exp()).sum::<f64>();
    logits
        .iter()
        .map(|logit| logit.exp() / logit_exp_sum)
        .collect()
}

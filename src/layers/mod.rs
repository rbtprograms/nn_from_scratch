struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    // NOTE: add grad part later in article
    weight_grad: Vec<Vec<f64>>,
    bias_grad: Vec<f64>
}

impl Layer {
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

        // map each output to a relu
        output.iter().map(|&x| relu(x)).collect()
    }

    // backward needs to return the error for this layer
    fn backward(&mut self, input: &Vec<f64>, output_grad: &Vec<f64>) -> Vec<f64>{
        let input_size = input.len();
        let output_size = self.biases.len();

        let mut output = vec![0.0; output_size];

        // want to find dE/dW for each weight, which is given by activation*output_error
        for i in 0..output_size {
            for j in 0..input_size {
                // updates each weight gradient, which we save on the struct
                self.weight_grad[i][j] = input[j] * output_grad[i];
                // summation for the error for each neuron in the layer
                output[j] += output_grad[i]*self.weights[i][j];
            }
            // bias gradient is just the output error
            self.bias_grad[i] = output_grad[i];
        }

        output
    }

    fn update_weights(&mut self, learning_rate: f64) {
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                self.weights[i][j] -= self.weight_grad[i][j]*learning_rate
            }
            self.biases[i] -= self.bias_grad[i]*learning_rate;
        }
    }
}

fn relu(x: f64) -> f64 {
    if x < 0.0 { 0.0 } else { x }
}
use rand::Rng;

pub struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
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
            let dot = weight.iter().zip(input).map(|(w, i)| w * i).sum::<f64>();

            pre_activations[i] = dot + self.biases[i];
            //want to map each weighted sum to the sigmoid activation
            post_activations[i] = sigmoid(pre_activations[i]);
        }
        (pre_activations, post_activations)
    }

    // backward needs to return the error for this layer
    pub fn backward(
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

    pub fn update_weights(&mut self, learning_rate: f64) {
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                self.weights[i][j] -= self.weight_grad[i][j] * learning_rate;
                self.weight_grad[i][j] = 0.0;
            }
            self.biases[i] -= self.bias_grad[i] * learning_rate;
            self.bias_grad[i] = 0.0;
        }
    }
}

fn sigmoid_prime(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

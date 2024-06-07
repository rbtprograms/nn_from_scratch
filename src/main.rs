// main.rs
use nn_from_scratch::NN;

fn main() {
    let x_train = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let y_train = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    let epochs = 100000;
    let learning_rate = 1.0;

    let data: Vec<(Vec<f64>, Vec<f64>)> = x_train
        .iter()
        .cloned()
        .zip(y_train.iter().cloned())
        .collect();

    // Create and configure the network
    let mut nn = NN::new();
    nn.add_layer(2, 3); // Input layer to hidden layer with 3 neurons
    nn.add_layer(3, 1); // Hidden layer to output layer with 1 neuron

    // Train the network
    nn.train(&data, epochs, learning_rate);

    // Test the network
    for input in x_train {
        let prediction = nn.predict(&input);
        println!("{:?} -> {:?}", input, prediction);
    }
}

// Definitions of Layer, NN, and other functions...

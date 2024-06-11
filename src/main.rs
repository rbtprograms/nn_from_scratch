// main.rs
use nn_from_scratch::network::NN;

use mnist::{Mnist, MnistBuilder};
use ndarray::Array2;

fn load_mnist(train_samps: usize, test_samps: usize) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .base_path("data")
        .label_format_digit()
        .training_set_length(train_samps.try_into().unwrap())
        .test_set_length(test_samps.try_into().unwrap())
        .finalize();

    // Convert the flattened vector of images to a 2D array (each row is an image)
    let trn_img = Array2::from_shape_vec((train_samps, 28 * 28), trn_img)
        .expect("Error converting training images")
        .mapv(|x| x as f64 / 255.0);
    let tst_img = Array2::from_shape_vec((test_samps, 28 * 28), tst_img)
        .expect("Error converting test images")
        .mapv(|x| x as f64 / 255.0);

    let trn_lbl = one_hot_encode(&trn_lbl, train_samps);
    let tst_lbl = one_hot_encode(&tst_lbl, test_samps);

    (trn_img, trn_lbl, tst_img, tst_lbl)
}

fn one_hot_encode(labels: &[u8], num_samples: usize) -> Array2<f64> {
    let mut one_hot = Array2::<f64>::zeros((num_samples, 10));
    for (i, &label) in labels.iter().enumerate() {
        one_hot[(i, label as usize)] = 1.0;
    }
    one_hot
}
fn main() {
    let train_samps: usize = 10000;
    let test_samps: usize = train_samps/10;
    // Load the MNIST dataset
    println!("Data Loading");
    let (trn_img, trn_lbl, tst_img, tst_lbl) = load_mnist(train_samps, test_samps);
    println!("Data Loaded");

    // Create and configure the network
    let mut nn = NN::new();
    nn.add_layer(784, 64);
    nn.add_layer(64, 32);
    nn.add_layer(32, 10);
    println!("Network ready");

    // Prepare the data
    let training_data: Vec<(Vec<f64>, Vec<f64>)> = trn_img
        .outer_iter()
        .zip(trn_lbl.outer_iter())
        .map(|(x, y)| (x.to_vec(), y.to_vec()))
        .collect();
    println!("Data ready");

    // Train the network
    println!("TRAINING STARTED");
    nn.train(&training_data, 10, 0.1); // Adjust epochs and learning rate as needed
    println!("TRAINING COMPLETE");

    // Evaluate the network
    let test_data: Vec<(Vec<f64>, Vec<f64>)> = tst_img
        .outer_iter()
        .zip(tst_lbl.outer_iter())
        .map(|(x, y)| (x.to_vec(), y.to_vec()))
        .collect();

    let mut correct = 0;
    let mut printed = 0;
    for (input, target) in test_data {
        let prediction = nn.predict(&input);
        let predicted_label = prediction
            .iter()
            .enumerate()
            .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap();

        let actual_label = target
            .iter()
            .enumerate()
            .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap();
        if printed < 5 {
            println!("predicted label: {}", predicted_label);
            println!("actual label: {}", actual_label);
            printed += 1;
        }

        if predicted_label == actual_label {
            correct += 1;
        }
    }

    println!("Accuracy: {}%", correct as f64 / test_samps as f64 * 100.0);
}

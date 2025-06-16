use ndarray_npy::read_npy;
use nalgebra::{DMatrix, DVector};
use std::error::Error;
struct Dense {
	weights: DMatrix<f32>,
	biases: DVector<f32>,
}

impl Dense {
	fn new(weights: DMatrix<f32>, biases: DVector<f32>) -> Self {
        Self { weights, biases }
    }

    fn forward(&self, input: &DVector<f32>) -> DVector<f32> {
        &self.weights * input + &self.biases
    }
}

pub struct Network {
	layers: Vec<Dense>,
}


fn load_weights(path: &str) -> Result<DMatrix<f32>, Box<dyn Error>> {
    let array: ndarray::Array2<f32> = read_npy(path)?;
    let (nrows, ncols) = array.dim();
    let data = array.as_slice().ok_or("Failed to get ndarray slice")?;
    Ok(DMatrix::from_row_slice(nrows, ncols, data))
}

fn load_biases(path: &str) -> Result<DVector<f32>, Box<dyn Error>> {
    let array: ndarray::Array1<f32> = read_npy(path)?;
    let data = array.as_slice().ok_or("Failed to get ndarray slice")?;
    Ok(DVector::from_row_slice(data))
}

impl Network {
	pub fn new(path: &str, num_layers: i32) -> Result<Self, Box<dyn Error>> {
		let mut network = Self { layers: Vec::new() };

		for i in 0..num_layers {
			let weights = load_weights(format!("{}layer_{}_weight.npy", path, i+1).as_str())?;
			let biases = load_biases(format!("{}layer_{}_bias.npy", path, i+1).as_str())?;
			let layer = Dense::new(weights, biases);
			network.layers.push(layer);
		}

		Ok(network)
	}

	pub fn forward(&self, mut x: DVector<f32>) -> DVector<f32> {
		for layer in self.layers.iter() {
			x = layer.forward(&x);
			x.apply(|a| *a = a.max(0.0)); // ReLU
		}
		x
	}
}


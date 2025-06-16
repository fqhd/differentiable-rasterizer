mod bezier;
mod circle;
mod network;
mod math;
pub use bezier::{Bezier, Vector2};

use crate::network::Network;

pub fn rasterize(curve: &Bezier, width: u32) -> Vec<f32> {
    let mut image = vec![1.0; (width * width) as usize];

    if let Ok(net) = Network::new("network/weights/", 5) {
        for j in 0..width {
            for i in 0..width {
                let index = j * width + i;
                let y = (j as f32) / (width as f32);
                let x = (i as f32) / (width as f32);
                image[index as usize] = curve.forward(x, y, &net);
            }
        }

        let inputs = nalgebra::DVector::from_vec(vec![0.05, 0.1, 0.3, 0.2, 0.5, 0.1, 0.6, 0.8]);
        let ans = net.forward(inputs)[0];
        println!("Ans: {}", ans);

    }



    image
}

pub fn optimize(curve: &mut Bezier, width: u32, target: &[f32], learning_rate: f32) -> f32 {
    curve.zero_grad();
    let mut loss = 0.0;

    for j in 0..width {
        for i in 0..width {
            let index = j * width + i;
            let label = target[index as usize];
            let y = (j as f32) / (width as f32);
            let x = (i as f32) / (width as f32);
            let (_, l) = curve.forward_with_gradients(x, y, label);
            loss += l;
        }
    }

    loss /= (width * width) as f32;

    curve.step(width, learning_rate);

    loss
}

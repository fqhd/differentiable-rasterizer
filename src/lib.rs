mod bezier;
mod circle;
pub use bezier::{Bezier, Vector2};

pub fn rasterize(curve: &Bezier, width: u32) -> Vec<f32> {
    let mut image = vec![1.0; (width * width) as usize];

    for j in 0..width {
        for i in 0..width {
            let index = j * width + i;
            let y = (j as f32) / (width as f32);
            let x = (i as f32) / (width as f32);
            image[index as usize] = curve.forward(x, y);
        }
    }

    image
}

pub fn optimize(
    curve: &mut Bezier,
    width: u32,
    raster: &Vec<f32>,
    target: &Vec<f32>,
    learning_rate: f32,
) -> f32 {
    curve.zero_grad();
    let mut loss = 0.0;

    for j in 0..width {
        for i in 0..width {
            let index = j * width + i;
            let c = raster[index as usize];
            let label = target[index as usize];
            let y = (j as f32) / (width as f32);
            let x = (i as f32) / (width as f32);
            loss += (label - c).powf(2.0);
            curve.backward(x, y, c, label);
        }
    }

    loss /= (width * width) as f32;

    curve.step(width, learning_rate);

    loss
}

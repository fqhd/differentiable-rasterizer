mod circle;
pub use circle::Circle;

pub fn rasterize(circle: &Circle, width: u32) -> Vec<f32> {
    let mut image = vec![0.0; (width * width * 3) as usize];

    for j in 0..width {
        for i in 0..width {
            let index = j * width * 3 + i * 3;
            let y = (j as f32) / (width as f32);
            let x = (i as f32) / (width as f32);
            let value = circle.forward(x, y);
            image[(index + 0) as usize] += value.0;
            image[(index + 1) as usize] += value.1;
            image[(index + 2) as usize] += value.2;
        }
    }

    image
}

pub fn optimize(
    circle: &mut Circle,
    width: u32,
    raster: &Vec<f32>,
    target: &Vec<f32>,
    learning_rate: f32,
) -> f32 {
    circle.zero_grad();
    let mut loss = 0.0;

    for j in 0..width {
        for i in 0..width {
            let index = j * width * 3 + i * 3;
            let color = (
                raster[(index + 0) as usize],
                raster[(index + 1) as usize],
                raster[(index + 2) as usize],
            );
            let label = (
                target[(index + 0) as usize],
                target[(index + 1) as usize],
                target[(index + 2) as usize],
            );

            let y = (j as f32) / (width as f32);
            let x = (i as f32) / (width as f32);

            loss += (label.0 - color.0).powf(2.0);
            loss += (label.1 - color.1).powf(2.0);
            loss += (label.2 - color.2).powf(2.0);

            circle.backward(x, y, color, label);
        }
    }

    loss /= (width * width * 3) as f32;

    circle.step(width, learning_rate);

    loss
}

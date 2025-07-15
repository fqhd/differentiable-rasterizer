mod circle;
pub use circle::Circle;

pub fn rasterize(circles: &Vec<Circle>, width: u32) -> Vec<f32> {
    let mut image = vec![0.0; (width * width * 3) as usize];

    for circle in circles {
        let radius = circle.z;

        let min_x = ((circle.x - radius) * width as f32).floor() as i32;
        let min_y = ((circle.y - radius) * width as f32).floor() as i32;

        let max_x = ((circle.x + radius) * width as f32).ceil() as i32 + 1;
        let max_y = ((circle.y + radius) * width as f32).ceil() as i32 + 1;

        let start_x = min_x.clamp(0, width as i32 - 1);
        let end_x = max_x.clamp(0, width as i32);

        let start_y = min_y.clamp(0, width as i32 - 1);
        let end_y = max_y.clamp(0, width as i32);

        for j in start_y..end_y {
            for i in start_x..end_x {
                let fx = i as f32 / width as f32;
                let fy = j as f32 / width as f32;

                let index = (j * width as i32 + i) * 3;
                let color = circle.forward(fx, fy);

                image[index as usize + 0] += color.0;
                image[index as usize + 1] += color.1;
                image[index as usize + 2] += color.2;
            }
        }
    }

    image
}

pub fn optimize(
    circles: &mut Vec<Circle>,
    width: u32,
    raster: &Vec<f32>,
    target: &Vec<f32>,
    learning_rate: f32,
    momentum: f32,
) -> f32 {
    for circle in circles.iter_mut() {
        circle.zero_grad();
    }
    let mut loss = 0.0;
    let mut total_loss = 0.0;

    for circle in circles.iter_mut() {
        let radius = circle.z;

        let min_x = ((circle.x - radius) * width as f32).floor() as i32;
        let min_y = ((circle.y - radius) * width as f32).floor() as i32;

        let max_x = ((circle.x + radius) * width as f32).ceil() as i32 + 1;
        let max_y = ((circle.y + radius) * width as f32).ceil() as i32 + 1;

        let start_x = min_x.clamp(0, width as i32 - 1);
        let end_x = max_x.clamp(0, width as i32);

        let start_y = min_y.clamp(0, width as i32 - 1);
        let end_y = max_y.clamp(0, width as i32);

        let mut total_gradient = 0.0;
        for j in start_y..end_y {
            for i in start_x..end_x {
                let index = j * width as i32 * 3 + i * 3;
                let fx = i as f32 / width as f32;
                let fy = j as f32 / width as f32;

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

                loss += (label.0 - color.0).powf(2.0);
                loss += (label.1 - color.1).powf(2.0);
                loss += (label.2 - color.2).powf(2.0);

                circle.backward(fx, fy, color, label);
                total_loss += 3.0;
                total_gradient += 1.0;
            }
        }
        circle.dx /= total_gradient;
        circle.dy /= total_gradient;
        circle.dz /= total_gradient;
        circle.dr /= total_gradient;
        circle.dg /= total_gradient;
        circle.db /= total_gradient;
    }

    loss /= total_loss;

    for circle in circles.iter_mut() {
        circle.step(learning_rate, momentum);
    }

    loss
}

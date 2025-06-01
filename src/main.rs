use differentiable_rasterizer::{Circle, compute_gradients, rasterize};
use image::{ImageBuffer, Rgb, RgbImage};

const WIDTH: u32 = 128;

fn main() {
    let my_circle_1 = Circle::new(0.3, 0.2, 0.2);
    let values_1 = rasterize(&my_circle_1, WIDTH);

    let mut img1: RgbImage = ImageBuffer::new(WIDTH, WIDTH);

    let mut value_iter = values_1.iter();

    for y in 0..WIDTH {
        for x in 0..WIDTH {
            let value = value_iter.next();
            if let Some(value) = value {
                let value = (value * 255.0) as u8;
                img1.put_pixel(x, y, Rgb([value; 3]));
            }
        }
    }

    let mut my_circle_2 = Circle::new(0.1, 0.2, 0.2);

    println!("{}", my_circle_1.x);
    println!("{}", my_circle_2.x);

    for i in 0..1000 {
        let values_2 = rasterize(&my_circle_2, WIDTH);
        let gradients = compute_gradients(&my_circle_2, WIDTH, &values_2, &values_1);
        my_circle_2.x += gradients.dx * 1e-3;
        println!("{}) Loss: {}", i, gradients.loss);
    }

    println!("{}", my_circle_2.x);

    // img.save("output.png").expect("Failed to save image");
}

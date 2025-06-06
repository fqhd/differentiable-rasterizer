use differentiable_rasterizer::{Bezier, Vector2, optimize, rasterize};
use image::{ImageBuffer, Rgb, RgbImage};
use std::fs::File;
use std::io::Write;

const WIDTH: u32 = 128;

fn main() -> Result<(), std::io::Error> {
    let target = get_target();

    save_image(&target, "target.png");

    let mut curve = Bezier::new(
        Vector2::new(0.0, 0.0),
        Vector2::new(0.6, 0.6),
        Vector2::new(1.0, 1.0),
    );
    let mut losses: Vec<f32> = Vec::new();
    let mut gradients: Vec<f32> = Vec::new();

    for i in 0..10 {
        let values = rasterize(&curve, WIDTH);
        let loss = optimize(&mut curve, WIDTH, &target, 1e-3);
        gradients.push(curve.da.x);
        println!("{}) Loss: {}", i, loss);
        losses.push(loss);
        let path = format!("frames/{}.png", i);
        save_image(&values, &path);
    }

    save_list(&losses, "losses.txt")?;
    save_list(&gradients, "gradients.txt")?;

    Ok(())
}

fn get_target() -> Vec<f32> {
    let curve = Bezier::new(
        Vector2::new(0.3, 0.8),
        Vector2::new(0.0, 0.0),
        Vector2::new(0.8, 0.3),
    );
    rasterize(&curve, WIDTH)
}

fn save_list(values: &[f32], path: &str) -> Result<(), std::io::Error> {
    let mut file = File::create(path)?;
    for v in values {
        writeln!(file, "{}", v)?;
    }
    Ok(())
}

fn save_image(values: &[f32], path: &str) {
    let mut image: RgbImage = ImageBuffer::new(WIDTH, WIDTH);

    let mut value_iter = values.iter();

    for y in 0..WIDTH {
        for x in 0..WIDTH {
            let value = value_iter.next();
            if let Some(value) = value {
                let value = (value * 255.0) as u8;
                image.put_pixel(x, y, Rgb([value; 3]));
            }
        }
    }

    image.save(path).expect("Failed to save image");
}

use differentiable_rasterizer::{Bezier, Vector2, optimize, rasterize};
use image::{ImageBuffer, Rgb, RgbImage};
use std::fs::File;
use std::io::Write;

const WIDTH: u32 = 256;

fn main() -> Result<(), std::io::Error> {
    let target = get_target();

    save_image(&target, "target.png");

    // let mut circle = Circle::new(0.1, 0.1, 0.2);
    // let mut losses: Vec<f32> = Vec::new();

    // for i in 0..200 {
    //     let values = rasterize(&circle, WIDTH);
    //     let loss = optimize(&mut circle, WIDTH, &values, &target, 1e-2);
    //     println!("{}) Loss: {}", i, loss);
    //     losses.push(loss);
    //     let path = format!("frames/{}.png", i);
    //     save_image(&values, &path);
    // }

    // // Write losses to disk
    // let mut file = File::create("losses.txt")?;
    // for l in losses {
    //     writeln!(file, "{}", l)?;
    // }

    Ok(())
}

fn get_target() -> Vec<f32> {
    let curve = Bezier::new(
        Vector2::new(0.6, 0.8),
        Vector2::new(0.0, 0.0),
        Vector2::new(0.8, 0.6),
    );
    let values = rasterize(&curve, WIDTH);
    values
}

fn save_image(values: &Vec<f32>, path: &str) {
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

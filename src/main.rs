use differentiable_rasterizer::{Circle, optimize, rasterize};
use image::{ImageBuffer, Rgb, RgbImage};
use std::fs::File;
use std::io::Write;

const WIDTH: u32 = 128;

fn main() -> Result<(), std::io::Error> {
    let target = get_target();

    save_image(&target, "target.png");

    let mut circle = Circle::new(0.4, 0.4, 0.1, 0.3, 0.5, 0.5);
    let mut losses: Vec<f32> = Vec::new();

    for i in 0..1000 {
        let values = rasterize(&circle, WIDTH);
        let loss = optimize(&mut circle, WIDTH, &values, &target, 2e-3);
        println!("{}) Loss: {}", i, loss);
        losses.push(loss);
        let path = format!("frames/{}.png", i);
        save_image(&values, &path);
    }

    // Write losses to disk
    let mut file = File::create("losses.txt")?;
    for l in losses {
        writeln!(file, "{}", l)?;
    }

    Ok(())
}

fn get_target() -> Vec<f32> {
    let circle = Circle::new(0.5, 0.5, 0.2, 1.0, 0.8, 0.5);
    let values = rasterize(&circle, WIDTH);
    values
}

fn save_image(values: &Vec<f32>, path: &str) {
    let mut image: RgbImage = ImageBuffer::new(WIDTH, WIDTH);

    let values: Vec<u8> = values.iter().map(|x| (x * 255.0) as u8).collect();

    for y in 0..WIDTH {
        for x in 0..WIDTH {
            let index = y * WIDTH * 3 + x * 3;
            let red = values[(index + 0) as usize];
            let green = values[(index + 1) as usize];
            let blue = values[(index + 2) as usize];
            image.put_pixel(x, y, Rgb([red, green, blue]));
        }
    }

    image.save(path).expect("Failed to save image");
}

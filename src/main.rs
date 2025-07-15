use differentiable_rasterizer::{Circle, optimize, rasterize};
use fastrand;
use image::{ImageBuffer, Rgb, RgbImage, imageops::FilterType};
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), std::io::Error> {
    let target = get_target(128);

    let mut circles = Vec::new();
    for _ in 0..1000 {
        circles.push(Circle::new(
            fastrand::f32(),
            fastrand::f32(),
            0.01 + fastrand::f32() * 0.1,
            fastrand::f32() * 0.1,
            fastrand::f32() * 0.1,
            fastrand::f32() * 0.1,
        ));
    }
    let mut losses = Vec::new();

    for i in 0..750 {
        let values = rasterize(&circles, 128);
        let loss = optimize(&mut circles, 128, &values, &target, 2e-3);
        println!("{}) Loss: {}", i, loss);
        losses.push(loss);
        let path = format!("frames/{}.png", i);

        let values = rasterize(&circles, 512);
        save_image(&values, &path, 512);
    }

    // Write losses to disk
    let mut file = File::create("losses.txt")?;
    for l in losses {
        writeln!(file, "{}", l)?;
    }

    Ok(())
}

fn get_target(width: u32) -> Vec<f32> {
    let img = image::open("cat.png")
        .expect("Failed to load image")
        .resize_exact(width, width, FilterType::Lanczos3)
        .to_rgb8();

    let mut values = Vec::with_capacity((width * width * 3) as usize);

    for pixel in img.pixels() {
        values.push(pixel[0] as f32 / 255.0);
        values.push(pixel[1] as f32 / 255.0);
        values.push(pixel[2] as f32 / 255.0);
    }

    values
}

fn save_image(values: &Vec<f32>, path: &str, width: u32) {
    let mut image: RgbImage = ImageBuffer::new(width, width);

    let values: Vec<u8> = values.iter().map(|x| (x * 255.0) as u8).collect();

    for y in 0..width {
        for x in 0..width {
            let index = y * width * 3 + x * 3;
            let red = values[(index + 0) as usize];
            let green = values[(index + 1) as usize];
            let blue = values[(index + 2) as usize];
            image.put_pixel(x, y, Rgb([red, green, blue]));
        }
    }

    image.save(path).expect("Failed to save image");
}

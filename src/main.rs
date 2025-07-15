use clap::Parser;
use differentiable_rasterizer::{Circle, optimize, rasterize};
use fastrand;
use image::{ImageBuffer, Rgb, RgbImage, imageops::FilterType};
use std::fs;
use std::path::Path;

/// A differentiable rasterization pipeline that can be used to compute the gradients of the MSE loss between a rasterized set of circles and a target image.
#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    /// Path to the input image (REQUIRED)
    file_path: String,

    /// Path to the output image (optional)
    #[arg(long, default_value_t = String::from("raster.png"))]
    output: String,

    /// Learning rate (optional)
    #[arg(long, default_value_t = 1e-3)]
    lr: f32,

    /// Momentum (optional)
    #[arg(long, default_value_t = 0.9)]
    momentum: f32,

    /// Width of the rasterized image (optional)
    #[arg(long, default_value_t = 512)]
    width: u32,

    /// Number of gradient descent steps to take (optional)
    #[arg(long, default_value_t = 1000)]
    n_iterations: u32,

    /// Number of circles (optional)
    #[arg(short = 'n', long = "n_circles", default_value_t = 100)]
    n_circles: u32,

    /// Enable verbose output
    #[arg(long, default_value_t = false)]
    verbose: bool,

    /// Enable verbose output
    #[arg(long, default_value_t = false)]
    save_frames: bool,
}

fn main() -> Result<(), std::io::Error> {
    let args = Args::parse();

    let target = get_target(&args.file_path, 128);

    let folder_path = "frames";

    if !Path::new(folder_path).exists() {
        fs::create_dir_all(folder_path)?;
    }

    let mut circles = Vec::new();

    for _ in 0..args.n_circles {
        circles.push(Circle::new(
            fastrand::f32(),
            fastrand::f32(),
            0.01 + fastrand::f32() * 0.1,
            fastrand::f32() * 0.1,
            fastrand::f32() * 0.1,
            fastrand::f32() * 0.1,
        ));
    }

    for i in 0..args.n_iterations {
        let values = rasterize(&circles, 128);
        let loss = optimize(&mut circles, 128, &values, &target, args.lr, args.momentum);
        if args.verbose {
            println!("{}) Loss: {}", i, loss);
        }
        if args.save_frames {
            let values = rasterize(&circles, args.width);
            let path = format!("frames/{}.png", i);
            save_image(&values, &path, args.width);
        }
    }

    let values = rasterize(&circles, args.width);
    save_image(&values, &args.output, args.width);

    Ok(())
}

fn get_target(file_path: &str, width: u32) -> Vec<f32> {
    let img = image::open(file_path)
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

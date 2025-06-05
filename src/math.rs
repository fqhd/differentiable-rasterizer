use std::{cmp::Ordering, f32::consts::PI};

use nalgebra::Vector2;

pub fn p_sigmoid(x: f32, offset: f32, sharpness: f32) -> f32 {
    1.0 / (1.0 + ((offset - x) * sharpness).exp())
}

pub fn dx_p_sigmoid(x: f32, offset: f32, sharpness: f32) -> f32 {
    if !(0.2..=0.5).contains(&x) {
        return 0.0;
    }
    let a = ((offset - x) * sharpness).exp();
    let t1 = 1.0 / (1.0 + a).powf(2.0);
    t1 * a * sharpness
}

pub fn distance(a: Vector2<f32>, b: Vector2<f32>) -> f32 {
    ((a.x - b.x).powf(2.0) + (a.y - b.y).powf(2.0)).sqrt()
}

pub fn dx_distance(a: Vector2<f32>, b: Vector2<f32>) -> f32 {
    (b.x - a.x) / distance(a, b)
}

pub fn dy_distance(a: Vector2<f32>, b: Vector2<f32>) -> f32 {
    (b.y - a.y) / distance(a, b)
}

pub fn solve_cubic(a: f32, b: f32, c: f32, d: f32) -> Vec<f32> {
    if a == 0.0 {
        panic!("Coefficient a cannot be zero for a cubic equation");
    }

    // Convert to depressed cubic: t^3 + pt + q = 0
    let a_inv = 1.0 / a;
    let b_over_3a = b * a_inv / 3.0;

    let p = (3.0 * a * c - b * b) / (3.0 * a * a);
    let q = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a);

    let discriminant = (q / 2.0).powi(2) + (p / 3.0).powi(3);
    let mut roots = Vec::new();

    match discriminant.total_cmp(&0.0) {
        Ordering::Greater => {
            // One real root
            let sqrt_disc = discriminant.sqrt();
            let u = (-q / 2.0 + sqrt_disc).cbrt();
            let v = (-q / 2.0 - sqrt_disc).cbrt();
            let t = u + v;
            roots.push(t - b_over_3a);
        }
        Ordering::Equal => {
            // Triple or double root
            let u = (-q / 2.0).cbrt();
            roots.push(2.0 * u - b_over_3a);
            roots.push(-u - b_over_3a);
        }
        Ordering::Less => {
            // Three real roots
            let r = (-p / 3.0).sqrt().powi(3);
            let phi = (-(q / 2.0) / r).acos();
            let two_sqrt_p3 = 2.0 * (-p / 3.0).sqrt();

            for k in 0..3 {
                let angle = (phi + 2.0 * PI * k as f32) / 3.0;
                let t = two_sqrt_p3 * angle.cos();
                roots.push(t - b_over_3a);
            }
        }
    }

    roots
}

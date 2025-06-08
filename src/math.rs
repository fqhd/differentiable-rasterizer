use std::{cmp::Ordering, f32::consts::PI};

use nalgebra::Vector2;

pub fn p_sigmoid(x: f32, offset: f32, sharpness: f32) -> f32 {
    1.0 / (1.0 + ((offset - x) * sharpness).exp())
}

pub fn dx_p_sigmoid(x: f32, offset: f32, sharpness: f32) -> f32 {
    if !(-0.2..=0.4).contains(&x) {
        return 0.0;
    }
    let a = ((offset - x) * sharpness).exp();
    let t1 = 1.0 / (1.0 + a).powf(2.0);
    t1 * a * sharpness
}

pub fn distance(a: Vector2<f32>, b: Vector2<f32>) -> f32 {
    squared_distance(a, b).sqrt()
}

pub fn squared_distance(a: Vector2<f32>, b: Vector2<f32>) -> f32 {
    (a.x - b.x).powf(2.0) + (a.y - b.y).powf(2.0)
}

pub fn dx_distance(a: Vector2<f32>, b: Vector2<f32>) -> f32 {
    (b.x - a.x) / distance(a, b)
}

pub fn dy_distance(a: Vector2<f32>, b: Vector2<f32>) -> f32 {
    (b.y - a.y) / distance(a, b)
}

pub fn solve_cubic(a: f32, b: f32, c: f32, d: f32) -> Vec<(f32, u8)> {
    assert_ne!(a, 0.0, "Coefficient a cannot be zero for a cubic equation");

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
            roots.push((t - b_over_3a, 0));
        }
        Ordering::Equal => {
            // Triple or double root
            let u = (-q / 2.0).cbrt();
            roots.push((2.0 * u - b_over_3a, 1));
            roots.push((-u - b_over_3a, 2));
        }
        Ordering::Less => {
            // Three real roots
            let r = (-p / 3.0).sqrt().powi(3);
            let phi = (-(q / 2.0) / r).acos();
            let two_sqrt_p3 = 2.0 * (-p / 3.0).sqrt();

            for k in 0..3 {
                let angle = (phi + 2.0 * PI * k as f32) / 3.0;
                let t = two_sqrt_p3 * angle.cos();
                roots.push((t - b_over_3a, k + 3));
            }
        }
    }

    roots
}

pub fn d_solve_cubic(
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    tag: u8,
    (da_dl, db_dl, dc_dl, dd_dl): (f32, f32, f32, f32),
) -> f32 {
    // The same starting stuff from `solve_cubic`
    assert_ne!(a, 0.0, "Coefficient a cannot be zero for a cubic equation");
    let a_inv = 1.0 / a;
    let a_inv_2 = a_inv * a_inv;
    let a_inv_4 = a_inv_2 * a_inv_2;
    let a_inv_6 = a_inv_4 * a_inv_2;

    let p_num = 3.0 * a * c - b * b;
    let q_num = 2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d;
    let p = p_num / (3.0 * a * a);
    let q = q_num / (27.0 * a * a * a);
    let disc = (q / 2.0).powi(2) + (p / 3.0).powi(3);
    let sqrtdisc = disc.sqrt();

    // --- Common derivatives
    // Derivative of p wrt. l
    let dp_dl =
        (3.0 * (c * da_dl + a * dc_dl) * a.powi(2) - 2.0 * p_num * a * da_dl) * a_inv_4 / 3.0;
    // Derivative of q wrt. l
    let dq_dl = ((6.0 * b * b * db_dl - 9.0 * (b * c * da_dl + a * c * db_dl + a * b * dc_dl)
        + 27.0 * (2.0 * a * d * da_dl + a * a * dd_dl))
        * a.powi(3)
        - 3.0 * q_num * a * a * da_dl)
        * a_inv_6
        / 27.0;
    // Derivative of discriminant wrt. l
    let ddisc_dl = q * dq_dl + p * p * dp_dl;
    // Derivative of -b / (3a)
    let delta = (a * db_dl - b * da_dl) * a_inv_2 / 3.0;
    // ---

    match tag {
        0 => {
            // u, v = cbrt(-q/2 +- sqrt(disc))
            // Derivative of 2 * sqrt(disc) wrt. l -- The factor of 2 comes from algebraic simplifications in the derivative
            let dsqrtdisc_dl = ddisc_dl / sqrtdisc;
            // Derivatives of 6 * u,v wrt. l -- The factor of 6 comes from taking out the common 1/6 factor in the derivatives of u, v
            let du_dl = (dq_dl + dsqrtdisc_dl) / (-q / 2.0 + sqrtdisc).powi(2).cbrt();
            let dv_dl = (dq_dl - dsqrtdisc_dl) / (-q / 2.0 - sqrtdisc).powi(2).cbrt();
            // Derivative of the root x = u + v - b/(3a) wrt. l
            (du_dl + dv_dl) / 6.0 + delta
        }
        1 => {
            // u = cbrt(-q/2)
            // Derivative of u wrt. l
            let du_dl = -dq_dl / (6.0 * (-q / 2.0).powi(2).cbrt());
            // Tag 1 corresponds to the root x = 2u - b/(3a)
            2.0 * du_dl + delta
        }
        2 => {
            // u = cbrt(-q/2)
            // Derivative of u wrt. l
            let du_dl = -dq_dl / (6.0 * (-q / 2.0).powi(2).cbrt());
            // Tag 2 corresponds to the root x = -u - b/(3a)
            -du_dl + delta
        }
        3..5 => {
            let r = (-p / 3.0).sqrt().powi(3);
            let phi = (-(q / 2.0) / r).acos();
            // Derivative of r wrt. l
            let dr_dl = -(-p / 3.0).sqrt() * dp_dl / 2.0;
            // Derivative of phi wrt. l
            let dphi_dl =
                (q * dr_dl - r * dq_dl) / (2.0 * r * r * (1.0 - (q / (2.0 * r)).powi(2)).sqrt());
            // The exact root can be extracted from the tag
            let k = tag - 3;

            // Derivative of x_k wrt. l -- Formula too big for me to want to write it here
            let angle = phi + 2.0 * PI * (k as f32) / 3.0;
            -((-3.0 / p).sqrt() * angle.cos() * dp_dl
                + 2.0 * (-p / 3.0).sqrt() * angle.sin() * dphi_dl)
                / 3.0
        }

        _ => panic!("Invalid tag for cubic root: {}", tag),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert_eq!(p_sigmoid(1.0, 0.4, 40.0), 1.0);
        assert!((p_sigmoid(0.5, 0.4, 40.0) - 0.982014).abs() < 0.00001)
    }

    #[test]
    fn test_dx_sigmoid() {
        assert_eq!(dx_p_sigmoid(0.4, 0.4, 40.0), 10.0);
        assert_eq!(dx_p_sigmoid(0.4, 0.4, 100.0), 25.0);
        assert!((dx_p_sigmoid(0.2, 0.4, 30.0) - 0.073995).abs() < 0.00001)
    }

    #[test]
    fn test_distance() {
        assert_eq!(
            distance(Vector2::new(0.0, 5.0), Vector2::new(0.0, 0.0)),
            5.0
        );
        assert!(
            (distance(Vector2::new(0.0, 5.0), Vector2::new(5.0, 0.0)) - 7.07107).abs() < 0.00001
        );
        assert!(
            (distance(Vector2::new(0.0, 5.0), Vector2::new(-10.0, 0.0)) - 11.18034).abs() < 0.00001
        );
    }

    #[test]
    fn test_dx_distance() {
        assert_eq!(
            dx_distance(Vector2::new(0.0, 5.0), Vector2::new(0.0, 0.0)),
            0.0
        );
        assert!(
            (dx_distance(Vector2::new(0.0, 5.0), Vector2::new(5.0, 0.0)) - 0.707107).abs()
                < 0.00001
        );
        assert!(
            (dx_distance(Vector2::new(0.0, 5.0), Vector2::new(-10.0, 0.0)) + 0.89443).abs()
                < 0.00001
        );
    }
}

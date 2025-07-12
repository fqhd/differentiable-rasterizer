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
    let g = b / (3.0 * a);

    let p = (3.0 * a * c - b * b) / (3.0 * a * a);
    let q = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a);

    let f = (q / 2.0).powi(2) + (p / 3.0).powi(3);
    let mut roots = Vec::new();

    match f.total_cmp(&0.0) {
        Ordering::Greater => {
            // One real root
            let u = (-q / 2.0 + f.sqrt()).cbrt();
            let v = (-q / 2.0 - f.sqrt()).cbrt();
            let t = u + v;
            roots.push((t - g, 0));
        }
        Ordering::Equal => {
            // Triple or double root
            let u = (-q / 2.0).cbrt();
            roots.push((2.0 * u - g, 1));
            roots.push((-u - g, 2));
        }
        Ordering::Less => {
            // Three real roots
            let r = (-p / 3.0).sqrt().powi(3);
            let l = (-(q / 2.0) / r).acos();
            let y = 2.0 * (-p / 3.0).sqrt();

            for k in 0..3 {
                let h = (l + 2.0 * PI * k as f32) / 3.0;
                let t = y * h.cos();
                roots.push((t - g, k + 3));
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
    (da_dx, db_dx, dc_dx, dd_dx): (f32, f32, f32, f32),
) -> f32 {
    // The same starting stuff from `solve_cubic`
    assert_ne!(a, 0.0, "Coefficient a cannot be zero for a cubic equation");
    let p = (3.0 * a * c - b * b) / (3.0 * a * a);
    let q = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a);
    let f = (q / 2.0).powi(2) + (p / 3.0).powi(3);

    let numerator = (3.0 * (da_dx * c + a * dc_dx) - 2.0 * b * db_dx) * 3.0 * a.powi(2)
        - (3.0 * a * c - b.powi(2)) * 6.0 * a * da_dx;
    let denominator = 9.0 * a.powi(4);
    let p_prime = numerator / denominator;
    let q_prime = (27.0
        * a.powi(3)
        * (6.0 * b.powi(2) * db_dx - 9.0 * (da_dx * b * c + a * db_dx * c + a * b * dc_dx)
            + 27.0 * (2.0 * a * da_dx * d + a.powi(2) * dd_dx))
        - 81.0 * a.powi(2) * da_dx * (2.0 * b.powi(3) - 9.0 * a * b * c + 27.0 * a.powi(2) * d))
        / (729.0 * a.powi(6));

    let f_prime = (q * q_prime) / 2.0 + (p.powi(2) * p_prime) / 9.0;

    let g_prime = ((a * db_dx) - (b * da_dx)) / (3.0 * a.powi(2));

    match tag {
        0 => {
            let u_base = -q / 2.0 + f.sqrt();
            let v_base = -q / 2.0 - f.sqrt();

            let u_base_prime = -q_prime / 2.0 + f_prime / (2.0 * f.sqrt());
            let v_base_prime = -q_prime / 2.0 - f_prime / (2.0 * f.sqrt());

            let u_prime = u_base_prime / (3.0 * u_base.cbrt().powi(2));
            let v_prime = v_base_prime / (3.0 * v_base.cbrt().powi(2));

            let t_prime = u_prime + v_prime;
            t_prime - g_prime
        }
        1 => {
            let u_prime = ((-q / 2.0).powf(-2.0 / 3.0) / 3.0) * (-q_prime / 2.0);
            2.0 * u_prime - g_prime
        }
        2 => {
            let u_prime = ((-q / 2.0).powf(-2.0 / 3.0) / 3.0) * (-q_prime / 2.0);
            -u_prime - g_prime
        }
        3..=5 => {
            let k = (tag - 3) as f32; // Map tag to k ∈ {0,1,2}
            let y_val = 2.0 * (-p / 3.0).sqrt(); // Recompute y as in solve_cubic
            let r_val = (-p / 3.0).powf(3.0 / 2.0); // r = (-p/3)^{3/2}
            let l_inner_val = -q / (2.0 * r_val); // l_inner = -q/(2r)
            let l_val = l_inner_val.acos(); // l = acos(l_inner)
            let h_angle_val = (l_val + 2.0 * PI * k) / 3.0; // h_angle = (l + 2πk)/3

            // Derivatives:
            let y_prime = -p_prime / (3.0 * (-p / 3.0).sqrt()); // FIXED sign
            let r_prime = -p_prime * (-p / 3.0).sqrt() / 2.0; // Correct as-is
            let l_inner_prime = (-q_prime) / (2.0 * r_val) + (q * r_prime) / (2.0 * r_val * r_val); // Correct as-is
            let l_prime = -l_inner_prime / (1.0 - l_inner_val.powi(2)).sqrt(); // FIXED sign
            let h_angle_prime = l_prime / 3.0; // dh_angle/dx = (1/3) dl/dx

            // Derivative of t = y * cos(h_angle):
            let t_prime = y_prime * h_angle_val.cos() - y_val * h_angle_val.sin() * h_angle_prime;
            t_prime - g_prime // Derivative of root (t - g)
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

    #[test]
    fn test_d_solve_cubic_1_root() {
        let mut x = 2.2;
        let dx = 1e-5;

        let a0 = 1.0 * x;
        let b0 = 5.0 * x;
        let c0 = 7.0;
        let d0 = 3.0;
        let roots0 = solve_cubic(a0, b0, c0, d0);

        x += dx;
        let a1 = 1.0 * x;
        let b1 = 5.0 * x;
        let c1 = 7.0;
        let d1 = 3.0;
        let roots1 = solve_cubic(a1, b1, c1, d1);

        for (i, (root, tag)) in roots0.into_iter().enumerate() {
            let truth = d_solve_cubic(a0, b0, c0, d0, tag, (1.0, 5.0, 0.0, 0.0));

            let approx = (roots1[i].0 - root) / dx;

            assert!(
                (approx - truth).abs() < 0.1,
                "Got different values... d_solve_cubic: {}, hard_approximation: {}",
                truth,
                approx
            );
        }
    }

    #[test]
    fn test_d_solve_cubic() {
        let mut x = 2.5;
        let dx = 1e-5;

        let a0 = 1.0 * x;
        let b0 = 3.0 * x;
        let c0 = 3.0;
        let d0 = -1.0;
        let roots0 = solve_cubic(a0, b0, c0, d0);

        x += dx;
        let a1 = 1.0 * x;
        let b1 = 3.0 * x;
        let c1 = 3.0;
        let d1 = -1.0;
        let roots1 = solve_cubic(a1, b1, c1, d1);

        for (i, (root, tag)) in roots0.into_iter().enumerate() {
            let truth = d_solve_cubic(a0, b0, c0, d0, tag, (1.0, 3.0, 0.0, 0.0));

            let approx = (roots1[i].0 - root) / dx;

            assert!(
                (approx - truth).abs() < 0.1,
                "Got different values... d_solve_cubic: {}, hard_approximation: {}, index: {}",
                truth,
                approx,
                i
            );
        }
    }
}
